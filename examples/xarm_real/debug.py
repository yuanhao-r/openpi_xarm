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
# CHECKPOINT_DIR = "/home/hil-serl/openpi_test/openpi/checkpoints/my_experiment/1000"
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
    # 保持 Mode 0 安全模式
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
    print("【单位修正版】Debug 模式启动")
    print("原理： 输入(Rad->Deg) -> 模型 -> 输出(Deg->Rad)")
    
    prompt = "pickup"

    try:
        while True:
            user_in = input("\n按回车预测 (q退出) > ")
            if user_in.strip() == 'q': break

            # --- A. 获取数据 ---
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
                cv2.waitKey(500) # 刷新窗口

            # --- B. 获取机械臂状态 (真实物理状态，弧度) ---
            code, current_joints_rad = arm.get_servo_angle(is_radian=True)
            if code != 0:
                current_joints_rad = [0.0] * 7
            else:
                # 补齐夹爪，假设当前是 0.0
                current_joints_rad = list(current_joints_rad[:6]) + [0.0] 
            
            # --- C. 【关键修正步骤 1】 输入转换：弧度 -> 度 ---
            # 只有这样，模型看到的数值范围才和训练时一致 (-180 ~ 180)
            current_joints_deg = np.array(current_joints_rad, dtype=np.float32)
            current_joints_deg[:6] = current_joints_deg[:6] * RAD_2_DEG # 转换前6轴
            
            # 放入 observation
            obs["state"] = current_joints_deg

            # --- D. 推理 ---
            example = {
                "cam_high": obs["cam_high"],
                "cam_left_wrist": obs["cam_left_wrist"],
                "cam_right_wrist": obs["cam_right_wrist"],
                "state": obs["state"], # 喂给模型的是【度】
                "prompt": prompt
            }
            
            start_t = time.time()
            result = policy.infer(example)
            infer_time = (time.time() - start_t) * 1000
            
            # 获取模型输出 (模型认为是【度】)
            pred_action_deg = np.array(result["actions"][0])

            # --- E. 【关键修正步骤 2】 输出转换：度 -> 弧度 ---
            pred_action_rad = pred_action_deg.copy()
            pred_action_rad[:6] = pred_action_rad[:6] * DEG_2_RAD # 转回弧度用于对比

            # --- F. 分析对比 (统一在弧度制下对比) ---
            current_joints_np = np.array(current_joints_rad)
            diff_rad = pred_action_rad - current_joints_np
            
            print(f"\n>>> 推理耗时: {infer_time:.1f}ms")
            print(f"{'JOINT':<8} | {'CURR(Rad)':<10} | {'PRED(Rad)':<10} | {'PRED(Deg)':<10} | {'DIFF(Rad)':<10}")
            print("-" * 75)
            
            joint_names = ["J1", "J2", "J3", "J4", "J5", "J6", "Gripper"]
            for i in range(7):
                c_rad = current_joints_np[i]
                p_rad = pred_action_rad[i]
                p_deg = pred_action_deg[i] # 看看模型原始输出是多少度
                d_val = diff_rad[i]
                
                marker = " << OK" if abs(d_val) < 0.15 else " << WARN"
                if i == 6: marker = "" 

                print(f"{joint_names[i]:<8} | {c_rad:10.4f} | {p_rad:10.4f} | {p_deg:10.1f} | {d_val:10.4f}{marker}")

            print("-" * 75)
            max_diff = np.max(np.abs(diff_rad[:6]))
            if max_diff < 0.2:
                print("✅ 验证成功！单位统一后，动作变得平滑。你可以直接修改控制脚本使用了。")
            else:
                print(f"⚠️ 依然存在较大跳变 ({max_diff:.4f} rad)。")
                print("可能性：1. 起始位置还是不对 2. 模型归一化参数(Mean/Std)因单位混乱彻底失效。")
                print("如果是后者，你必须重训。")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        arm.disconnect()

if __name__ == "__main__":
    main()