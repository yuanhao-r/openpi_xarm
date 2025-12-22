import time
import cv2
import numpy as np
import jax
from xarm.wrapper import XArmAPI

# OpenPI 依赖
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools 

# ================= 配置区域 =================
ROBOT_IP = "192.168.1.232"
CONFIG_NAME = "pi05_xarm"
# 请确认这是你最新的、用【弧度】数据训练的 checkpoint
CHECKPOINT_DIR = "/home/hil-serl/openpi_test/openpi/checkpoints/1206night/19999" 

CAMERAS = {
    "cam_high": 4,
    "cam_left_wrist": 0,
    "cam_right_wrist": 2
}
# ===========================================

def main():
    # 设置打印格式，不使用科学计数法，保留4位小数
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    print(f"Loading Model: {CONFIG_NAME}...")
    try:
        config = _config.get_config(CONFIG_NAME)
        policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
        print(">>> Model Loaded Successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Connecting to Robot (Safe Mode)...")
    arm = XArmAPI(ROBOT_IP)
    arm.clean_error()
    arm.clean_warn()
    # 保持 Mode 0 (位置模式)，但不发送指令，这样最安全
    arm.set_mode(0)
    arm.set_state(0)
    
    caps = {}
    print("Opening Cameras...")
    for name, idx in CAMERAS.items():
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            caps[name] = cap
    print(f"Cameras opened: {list(caps.keys())}")

    print("-" * 60)
    print("【调试模式启动 - 纯弧度版】")
    print("本程序不控制机械臂。仅对比【当前状态】与【模型预测】。")
    print("按 'Enter' 键刷新一次预测，按 'q' 退出。")
    print("-" * 60)
    
    prompt = "pick object" 

    try:
        while True:
            user_in = input("\n按回车进行一次预测 (输入 q 退出) > ")
            if user_in.strip().lower() == 'q': break

            # --- A. 获取图像数据 ---
            obs = {}
            for name, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = image_tools.resize_with_pad(frame_rgb, 224, 224)
                    obs[name] = image_tools.convert_to_uint8(frame_resized)
                else:
                    print(f"[Warn] Camera {name} read failed")
                    obs[name] = np.zeros((224,224,3), dtype=np.uint8)

            # --- B. 获取机械臂状态 (真实物理状态，弧度) ---
            code, current_joints_rad = arm.get_servo_angle(is_radian=True)
            if code != 0:
                print(f"[Error] Get joint failed: {code}")
                current_joints_rad = [0.0] * 7
            else:
                # 补齐夹爪: 假设读回来的是6维，手动补第7维
                # 注意：这里我们只能拿到关节，拿不到真实的夹爪状态，暂时补0.0或者上次的状态
                current_joints_rad = list(current_joints_rad[:6]) + [0.0] 
            
            # --- C. 构造 State (纯弧度) ---
            # 直接使用弧度，无需转换
            current_state_vec = np.array(current_joints_rad, dtype=np.float32)
            obs["state"] = current_state_vec

            # --- D. 推理 ---
            example = {
                "cam_high": obs["cam_high"],
                "cam_left_wrist": obs["cam_left_wrist"],
                "cam_right_wrist": obs["cam_right_wrist"],
                "state": obs["state"], 
                "prompt": prompt
            }
            
            start_t = time.time()
            result = policy.infer(example)
            infer_time = (time.time() - start_t) * 1000
            
            # 获取模型输出 Chunk (模型输出也是弧度)
            # 形状通常是 (chunk_size, 7)
            pred_actions_chunk = np.array(result["actions"])
            
            # 我们只看第一步 (Next Step)
            pred_action_rad = pred_actions_chunk[0]

            # --- E. 结果分析与打印 ---
            
            # 1. 拆分数据
            # 关节角度 (前6维)
            joints_rad_out = pred_action_rad[:6]
            # 夹爪状态 (第7维)
            gripper_out = pred_action_rad[6]

            # 2. 计算差异 (Diff)
            # 都是弧度，直接减
            diff_rad = joints_rad_out - np.array(current_joints_rad[:6])

            print(f"\n[推理耗时: {infer_time:.1f}ms] | Prompt: {prompt}")
            print("=" * 90)
            # 表头去掉了 Deg 列，只保留 Rad
            print(f"{'Index':<6} | {'Name':<10} | {'Current(Rad)':<14} | {'ModelOut(Rad)':<14} | {'Diff(Rad)':<12}")
            print("-" * 90)
            
            joint_names = ["J1", "J2", "J3", "J4", "J5", "J6"]
            
            # 打印 6 个关节
            for i in range(6):
                curr = current_joints_rad[i]
                mod_rad = joints_rad_out[i]
                diff = diff_rad[i]
                
                # 如果差值很大 (> 0.1 弧度，约 5.7度)，标记一下 warning
                marker = "⚠️" if abs(diff) > 0.1 else "" 
                
                print(f"{i:<6} | {joint_names[i]:<10} | {curr:14.4f} | {mod_rad:14.4f} | {diff:12.4f} {marker}")
            
            print("-" * 90)
            
            # 打印夹爪
            print(f"{6:<6} | {'GRIPPER':<10} | {'N/A':<14} | {gripper_out:14.4f} | {'N/A':<12}")
            
            if gripper_out > 0.8:
                print(">>> �� 模型意图: [闭合夹爪] (Close)")
            elif gripper_out < 0.2:
                print(">>> ⚪ 模型意图: [张开夹爪] (Open)")
            else:
                print(f">>> �� 模型意图: [中间状态] (Value: {gripper_out:.4f})")
            
            print("=" * 90)
            
            # 打印整个 chunk 的夹爪趋势
            gripper_trajectory = pred_actions_chunk[:, 6]
            print(f"未来 {len(gripper_trajectory)} 步夹爪趋势: {gripper_trajectory}")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        arm.disconnect()

if __name__ == "__main__":
    main()