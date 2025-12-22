import time
import cv2
import numpy as np
import jax
from xarm.wrapper import XArmAPI
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

# ================= 配置区域 =================
ROBOT_IP = "192.168.1.232"
CONFIG_NAME = "pi05_xarm"
# 请确保路径正确
CHECKPOINT_DIR = "/home/hil-serl/openpi_test/openpi/checkpoints/19999"

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
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    # ... (模型加载和机器人连接代码保持不变) ...
    print(f"Loading Model: {CONFIG_NAME}...")
    try:
        config = _config.get_config(CONFIG_NAME)
        policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
        print(">>> Model Loaded Successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Connecting to Robot...")
    arm = XArmAPI(ROBOT_IP)
    arm.clean_error()
    arm.clean_warn()
    arm.set_mode(0)
    arm.set_state(0)
    
    caps = {}
    for name, idx in CAMERAS.items():
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            caps[name] = cap

    print("-" * 60)
    print("【夹爪 Debug 模式】连续预测启动")
    prompt = "pickup"

    # 设置连续预测的步数
    NUM_STEPS = 10 

    try:
        user_in = input(f"\n按回车执行 {NUM_STEPS} 步连续预测 (q退出) > ")
        if user_in.strip() == 'q': return

        for step in range(NUM_STEPS):
            print(f"\n--- STEP {step + 1}/{NUM_STEPS} ---")

            # --- A. 获取数据 (图像) ---
            obs = {}
            display_imgs = []
            for name, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    display_imgs.append(cv2.resize(frame, (320, 240)))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = image_tools.resize_with_pad(frame_rgb, 224, 224)
                    obs[name] = image_tools.convert_to_uint8(frame_resized)
                else:
                    obs[name] = np.zeros((224,224,3), dtype=np.uint8)
            
            if display_imgs:
                cv2.imshow("Debug View", np.hstack(display_imgs))
                cv2.waitKey(1) 

            # --- B. 获取机械臂状态 (真实物理状态，弧度) ---
            code, current_joints_rad = arm.get_servo_angle(is_radian=True)
            if code != 0:
                current_joints_rad = [0.0] * 7
            else:
                # 补齐夹爪，假设当前是 0.0 (这里假定夹爪状态也是弧度，但实际上它可能是归一化值)
                current_joints_rad = list(current_joints_rad[:6]) + [0.0] 
            
            # --- C. 输入转换：弧度 -> 度 ---
            current_joints_deg = np.array(current_joints_rad, dtype=np.float32)
            current_joints_deg[:6] = current_joints_deg[:6] * RAD_2_DEG # 转换前6轴
            # 夹爪 (索引 6) 保持不变，因为它在训练时可能是归一化的 [0, 1]
            obs["state"] = current_joints_deg

            # --- D. 推理 ---
            example = {
                "cam_high": obs["cam_high"],
                "cam_left_wrist": obs["cam_left_wrist"],
                "cam_right_wrist": obs["cam_right_wrist"],
                "state": obs["state"], # 喂给模型的是【度】+【夹爪归一化值】
                "prompt": prompt
            }
            
            start_t = time.time()
            result = policy.infer(example)
            infer_time = (time.time() - start_t) * 1000
            pred_action_deg = np.array(result["actions"][0]) # 模型输出的 7D 动作 (度 + 夹爪值)

            # --- E. 输出转换：度 -> 弧度 ---
            pred_action_rad = pred_action_deg.copy()
            pred_action_rad[:6] = pred_action_rad[:6] * DEG_2_RAD # 关节转回弧度

            # --- F. 分析对比 ---
            current_joints_np = np.array(current_joints_rad)
            diff_rad = pred_action_rad - current_joints_np
            
            print(f">>> 推理耗时: {infer_time:.1f}ms")
            print(f"{'JOINT':<8} | {'CURR(Rad)':<10} | {'PRED(Rad)':<10} | {'PRED(Deg)':<10} | {'DIFF(Rad)':<10}")
            print("-" * 75)
            
            joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "Gripper"]
            for i in range(7):
                c_rad = current_joints_np[i]
                p_rad = pred_action_rad[i]
                p_deg = pred_action_deg[i]
                d_val = diff_rad[i]
                
                marker = " << OK" if abs(d_val) < 0.15 else " << WARN"
                if i == 6: marker = " << CLAMP" # 夹爪标记
                
                print(f"{joint_names[i]:<8} | {c_rad:10.4f} | {p_rad:10.4f} | {p_deg:10.1f} | {d_val:10.4f}{marker}")

            print("-" * 75)
            # 关键的夹爪诊断信息
            gripper_action_value = pred_action_deg[6]
            print(f"⭐ 夹爪 ACTION 原始输出值: {gripper_action_value:.4f}")
            
            if gripper_action_value > 0.9:
                 print("⚠️ 夹爪值接近 1.0 (最大值)，通常表示【完全打开】。")
            elif gripper_action_value < 0.1:
                 print("✅ 夹爪值接近 0.0 (最小值)，通常表示【完全关闭】。")
            else:
                 print("ℹ️ 夹爪值处于中间态，可能是过渡状态。")
            
            # --- 动作执行 (可选，但推荐) ---
            # 如果你想要机器人实际移动，需要在这里添加 xarm 的动作执行逻辑
            # time.sleep(1 / 10) # 模拟帧率
            
            max_diff = np.max(np.abs(diff_rad[:6]))
            if max_diff < 0.005:
                print("✨ 预测动作极小，可能已到达目标点，提前结束连续预测。")
                break
            
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        arm.disconnect()

if __name__ == "__main__":
    main()