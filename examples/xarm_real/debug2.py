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
# CHECKPOINT_DIR = "/home/hil-serl/openpi_test/openpi/checkpoints/my_experiment/1000" 
# CHECKPOINT_DIR = "/home/hil-serl/openpi_test/openpi/checkpoints/19999" 
CHECKPOINT_DIR = "/home/hil-serl/openpi_test/openpi/checkpoints/1206night/19999" 

CAMERAS = {
    "cam_high": 4,
    "cam_left_wrist": 0,
    "cam_right_wrist": 2
}

# 转换常量
RAD_2_DEG = 180.0 / np.pi
DEG_2_RAD = np.pi / 180.0
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
    print("【调试模式启动】")
    print("本程序不控制机械臂。仅对比【当前状态】与【模型预测】。")
    print("按 'Enter' 键刷新一次预测，按 'q' 退出。")
    print("-" * 60)
    
    prompt = "pick object" # 或者你之前训练时的指令

    try:
        while True:
            user_in = input("\n按回车进行一次预测 (输入 q 退出) > ")
            if user_in.strip().lower() == 'q': break

            # --- A. 获取图像数据 ---
            obs = {}
            for name, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    # 显示一下图像，确认摄像头没黑屏
                    # cv2.imshow(name, cv2.resize(frame, (320, 240)))
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = image_tools.resize_with_pad(frame_rgb, 224, 224)
                    obs[name] = image_tools.convert_to_uint8(frame_resized)
                else:
                    print(f"[Warn] Camera {name} read failed")
                    obs[name] = np.zeros((224,224,3), dtype=np.uint8)
            
            # cv2.waitKey(10) # 刷新图像窗口

            # --- B. 获取机械臂状态 (真实物理状态，弧度) ---
            code, current_joints_rad = arm.get_servo_angle(is_radian=True)
            if code != 0:
                print(f"[Error] Get joint failed: {code}")
                current_joints_rad = [0.0] * 7
            else:
                # 补齐夹爪: 如果没有真实夹爪反馈，暂时用 self.current_gripper_state 或者 0.0
                # 这里我们假设读回来的是6维，补第7维
                current_joints_rad = list(current_joints_rad[:6]) + [0.0] 
            
            # --- C. 输入转换：弧度 -> 度 ---
            # 这是喂给模型的 current state
            current_joints_deg = np.array(current_joints_rad, dtype=np.float32)
            current_joints_deg[:6] = current_joints_deg[:6] * RAD_2_DEG 
            
            obs["state"] = current_joints_deg

            # --- D. 推理 ---
            example = {
                "cam_high": obs["cam_high"],
                "cam_left_wrist": obs["cam_left_wrist"],
                "cam_right_wrist": obs["cam_right_wrist"],
                "state": obs["state"], # 度
                "prompt": prompt
            }
            
            start_t = time.time()
            result = policy.infer(example)
            infer_time = (time.time() - start_t) * 1000
            
            # 获取模型输出 Chunk (假设模型输出的是【度】)
            # 形状通常是 (chunk_size, 7)
            pred_actions_chunk = np.array(result["actions"])
            
            # 我们只看第一步 (Next Step)
            pred_action_deg = pred_actions_chunk[0]

            # --- E. 结果分析与打印 ---
            
            # 1. 拆分数据
            # 关节角度 (前6维)
            joints_deg_out = pred_action_deg[:6]
            # 夹爪状态 (第7维)
            gripper_out = pred_action_deg[6]

            # 2. 计算差异 (Diff)
            # 为了方便对比，我们把模型输出转回弧度，和当前真实弧度做减法
            joints_rad_out = joints_deg_out * DEG_2_RAD
            diff_rad = joints_rad_out - np.array(current_joints_rad[:6])

            print(f"\n[推理耗时: {infer_time:.1f}ms] | Prompt: {prompt}")
            print("=" * 90)
            print(f"{'Index':<6} | {'Name':<10} | {'Current(Rad)':<12} | {'ModelOut(Deg)':<14} | {'ModelOut(Rad)':<14} | {'Diff(Rad)':<12}")
            print("-" * 90)
            
            joint_names = ["J1", "J2", "J3", "J4", "J5", "J6"]
            
            # 打印 6 个关节
            for i in range(6):
                curr = current_joints_rad[i]
                mod_deg = joints_deg_out[i]
                mod_rad = joints_rad_out[i]
                diff = diff_rad[i]
                
                # 如果差值很大，标记一下 warning
                marker = "⚠️" if abs(diff) > 0.1 else "" 
                
                print(f"{i:<6} | {joint_names[i]:<10} | {curr:12.4f} | {mod_deg:14.4f} | {mod_rad:14.4f} | {diff:12.4f} {marker}")
            
            print("-" * 90)
            
            # 打印夹爪 (单独处理)
            # 夹爪不需要转弧度，它通常是 0.0 ~ 1.0
            print(f"{6:<6} | {'GRIPPER':<10} | {'N/A':<12} | {gripper_out:14.4f} | {'(raw val)':<14} | {'N/A':<12}")
            
            if gripper_out > 0.8:
                print(">>> �� 模型意图: [闭合夹爪] (Close)")
            elif gripper_out < 0.2:
                print(">>> �� 模型意图: [张开夹爪] (Open)")
            else:
                print(f">>> �� 模型意图: [中间状态/无动作] (Value: {gripper_out:.4f})")
            
            print("=" * 90)
            
            # 如果想看整个 chunk 的夹爪趋势，可以打印出来
            gripper_trajectory = pred_actions_chunk[:, 6]
            print(f"未来 {len(gripper_trajectory)} 步夹爪趋势: {gripper_trajectory}")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # cv2.destroyAllWindows()
        arm.disconnect()

if __name__ == "__main__":
    main()