import time
import cv2
import numpy as np
import jax
import sys
import os
import termios
import tty
import select
import threading

from xarm.wrapper import XArmAPI

# -----------------------------------------------------------------------------
# 路径设置
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
openpi_client_path = os.path.join(current_dir, "../../packages/openpi-client/src")
sys.path.append("/home/openpi/src")
sys.path.append(os.path.abspath(openpi_client_path))

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

# -----------------------------------------------------------------------------
# 1. 配置区域
# -----------------------------------------------------------------------------
ROBOT_IP = "192.168.1.232"
CONFIG_NAME = "pi05_xarm_1212_night" 
CHECKPOINT_DIR = "/home/openpi/checkpoints/exp11/30000"
CAMERAS = {
    "cam_left_wrist": 0,
    "cam_right_wrist": 2
}
CROP_CONFIGS = {
    "cam_left_wrist": (118, 60, 357, 420),
    "cam_right_wrist": (136, 57, 349, 412)
}

CONTROL_FREQ = 10 # 10Hz
EXECUTE_STEPS = 1 
JOINT_LIMITS = [
    (-6.2, 6.2), (-2.0, 2.0), (-2.9, 2.9), 
    (-3.1, 3.1), (-1.6, 1.8), (-6.2, 6.2)
]
HOME_POS = [539.120605, 17.047951, 100-59.568863, 3.12897, 0.012689, -1.01436]
MIN_SAFE_Z = -69
SLOW_DOWN_FACTOR = 3.0  
INTERPOLATION_FREQ = 100.0 

# -----------------------------------------------------------------------------
# 工具类：非阻塞键盘检测 (Linux)
# -----------------------------------------------------------------------------
class KeyPoller:
    def __enter__(self):
        # 保存终端设置
        self.fd = sys.stdin.fileno()
        self.old_term = termios.tcgetattr(self.fd)
        # 设置为 Raw 模式，读取单个字符不需要回车
        new_term = termios.tcgetattr(self.fd)
        new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(self.fd, termios.TCSANOW, new_term)
        return self

    def __exit__(self, type, value, traceback):
        # 恢复终端设置
        termios.tcsetattr(self.fd, termios.TCSANOW, self.old_term)

    def poll(self):
        # 检查是否有输入
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        if not dr:
            return None
        return sys.stdin.read(1)

# -----------------------------------------------------------------------------
# 2. 硬件封装类 (保持原样，略微优化)
# -----------------------------------------------------------------------------
class XArmHardware:
    def __init__(self, ip, camera_indices):
        print(f"Connecting to xArm at {ip}...")
        self.arm = XArmAPI(ip)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_tgpio_modbus_baudrate(baud=115200)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        
        self.caps = {}
        for name, idx in camera_indices.items():
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.caps[name] = cap
            else:
                print(f"[Warn] Failed to open camera {name}")
        
        self.current_gripper_state = 0.0
        self.open_gripper()
        time.sleep(1.0) 

    def close_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 1.0

    def open_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 0.0

    def get_observation(self) -> dict:
        obs = {}
        for name, cap in self.caps.items():
            ret, frame = cap.read()
            if not ret: frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if name in CROP_CONFIGS:
                x, y, w, h = CROP_CONFIGS[name]
                frame = frame[y:y+h, x:x+w]
            obs[name] = image_tools.convert_to_uint8(frame)
        
        obs["cam_high"] = np.zeros((224, 224, 3), dtype=np.uint8) 

        code, joints_rad = self.arm.get_servo_angle(is_radian=True)
        if code != 0: joints_rad = [0.0] * 7
        
        state_vec = np.array(joints_rad[:6], dtype=np.float32)
        state_vec_final = np.append(state_vec, self.current_gripper_state)
        obs["state"] = state_vec_final
        return obs

    def move_to_start(self, target_action_rad):
        print(">>> Moving to inference start position...")
        target_joints = target_action_rad[:6]
        safe_joints = [np.clip(angle, low, high) for angle, (low, high) in zip(target_joints, JOINT_LIMITS)]
        
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_servo_angle(angle=safe_joints, speed=0.35, is_radian=True, wait=True)
        
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.5)

    def execute_action(self, action_rad):
        target_joints_rad = action_rad[:6]
        target_gripper = action_rad[6]

        safe_joints = [np.clip(angle, low, high) for angle, (low, high) in zip(target_joints_rad, JOINT_LIMITS)]

        # Z轴保护
        ret, pose = self.arm.get_forward_kinematics(angles=safe_joints, input_is_radian=True, return_is_radian=True)
        if ret == 0:
            pred_z = pose[2]
            if pred_z < MIN_SAFE_Z:
                safe_pose = list(pose)
                safe_pose[2] = MIN_SAFE_Z
                ret_ik, ik_joints = self.arm.get_inverse_kinematics(safe_pose, input_is_radian=True, return_is_radian=True)
                if ret_ik == 0: safe_joints = list(ik_joints)
                else: return

        # 插值平滑
        code, current_joints = self.arm.get_servo_angle(is_radian=True)
        if code != 0: return
        curr_j = np.array(current_joints[:6])
        targ_j = np.array(safe_joints[:6])
        
        duration = (1.0 / CONTROL_FREQ) * SLOW_DOWN_FACTOR
        steps = int(duration * INTERPOLATION_FREQ)
        if steps < 1: steps = 1
        
        for i in range(1, steps + 1):
            alpha = i / steps
            interp = curr_j + (targ_j - curr_j) * alpha
            full = np.append(interp, 0.0)
            self.arm.set_servo_angle_j(angles=full, is_radian=True)
            time.sleep(1.0 / INTERPOLATION_FREQ)

        if target_gripper > 0.8: self.close_gripper()
        elif target_gripper < 0.2: self.open_gripper()
    
    def move_home_scripted(self):
        print(">>> Returning to Home...")
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_position(
            x=HOME_POS[0], y=HOME_POS[1], z=HOME_POS[2],
            roll=HOME_POS[3], pitch=HOME_POS[4], yaw=HOME_POS[5],
            speed=100, wait=True, is_radian=True
        )
    def move_DownOfHome_scripted(self):
        print(">>> Returning to Home...")
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_position(
            x=HOME_POS[0], y=HOME_POS[1], z=HOME_POS[2]-100,
            roll=HOME_POS[3], pitch=HOME_POS[4], yaw=HOME_POS[5],
            speed=100, wait=True, is_radian=True
        )

    def close(self):
        self.arm.set_mode(0)
        self.arm.set_state(0)
        for cap in self.caps.values(): cap.release()
        self.arm.disconnect()

# -----------------------------------------------------------------------------
# 3. 主程序逻辑
# -----------------------------------------------------------------------------
def main():
    # A. 加载模型 (只加载一次)
    print(f"Loading Config: {CONFIG_NAME}...")
    config = _config.get_config(CONFIG_NAME)
    print(f"Loading Checkpoint: {CHECKPOINT_DIR}...")
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    print("Model Loaded.")

    # B. 初始化硬件 (只初始化一次)
    robot = XArmHardware(ROBOT_IP, CAMERAS)
    prompt = "pick up the industrial components A"
    
    try:
        episode_count = 0
        
        # === 外层循环：Episode 管理 ===
        while True:
            episode_count += 1
            print(f"\n" + "="*50)
            print(f"Ready for Episode {episode_count}")
            print(f"Instructions:")
            print(f"  [ENTER] : Start Inference")
            print(f"  ['o']   : Emergency Reset (Open & Home)")
            print(f"  [Ctrl+C]: Exit Program")
            print("="*50)

            # --- 等待用户回车开始 ---
            # 为了在等待回车时也能响应 'o'，我们需要循环检测
            print(">>> Waiting for input...", end="", flush=True)
            start_episode = False
            
            # 使用 Raw 模式监听按键
            with KeyPoller() as key_poller:
                while True:
                    key = key_poller.poll()
                    if key is not None:
                        if key == '\n' or key == '\r': # Enter
                            start_episode = True
                            print("\n>>> Starting...")
                            break
                        elif key == 'o': # Reset
                            print("\n>>> Reset Requested.")
                            robot.move_home_scripted()
                            robot.move_DownOfHome_scripted()
                            robot.open_gripper()
                            robot.move_home_scripted()
                            print("\n>>> Reset Done. Ready.")
                            # 继续等待回车
                        elif key == '\x03': # Ctrl+C
                            raise KeyboardInterrupt
                    time.sleep(0.01)

            if not start_episode: continue

            # --- 1. 对齐阶段 (Pre-inference) ---
            print("[Phase 1] Aligning to start position...")
            # 在移动前也检测一次 'o'
            with KeyPoller() as key_poller:
                if key_poller.poll() == 'o':
                    robot.open_gripper()
                    robot.move_home_scripted()
                    continue

            raw_obs = robot.get_observation()
            example = {
                "cam_left_wrist": raw_obs["cam_left_wrist"],
                "cam_right_wrist": raw_obs["cam_right_wrist"],
                "state": raw_obs["state"],
                "prompt": prompt
            }
            result = policy.infer(example)
            start_action_rad = np.array(result["actions"])[0]
            robot.move_to_start(start_action_rad)

            # --- 2. 推理控制循环 ---
            print("[Phase 2] AI Control Loop (Press 'o' to abort)...")
            episode_success = False
            aborted = False

            # 使用 KeyPoller 进入实时控制循环
            with KeyPoller() as key_poller:
                while True:
                    # >>> 实时检测 'o' 键 <<<
                    key = key_poller.poll()
                    if key == 'o':
                        print("\n" + "!"*40)
                        print(">>> EMERGENCY ABORT ('o' pressed)")
                        print("!"*40)
                        aborted = True
                        break
                    
                    # 1. 观测
                    raw_obs = robot.get_observation()
                    example = {
                        "cam_left_wrist": raw_obs["cam_left_wrist"],
                        "cam_right_wrist": raw_obs["cam_right_wrist"],
                        "state": raw_obs["state"],
                        "prompt": prompt
                    }

                    # 2. 推理
                    result = policy.infer(example)
                    action_chunk = np.array(result["actions"]) 

                    # 3. 终止检测 (夹爪检测)
                    check_steps = 1
                    gripper_preds = action_chunk[:check_steps, 6]
                    if np.any(gripper_preds > 0.8):
                        print(f"\n>>> Grasp detected (Val: {gripper_preds[0]:.2f}). Stopping.")
                        episode_success = True
                        break

                    # 4. 执行动作 (分步执行以保持控制频率)
                    steps_to_run = min(EXECUTE_STEPS, len(action_chunk))
                    for i in range(steps_to_run):
                        step_start = time.time()
                        
                        # 在动作执行微步中也检测 'o'，反应更灵敏
                        if key_poller.poll() == 'o':
                            aborted = True; break
                            
                        robot.execute_action(action_chunk[i])
                        
                        elapsed = time.time() - step_start
                        sleep_time = (1.0 / CONTROL_FREQ) - elapsed
                        if sleep_time > 0: time.sleep(sleep_time)
                    
                    if aborted: break

            # --- 3. 回合结束处理 ---
            if aborted:
                # 按下 'o'：张开夹爪，回原点
                robot.open_gripper()
                time.sleep(0.5)
                robot.move_home_scripted()
            elif episode_success:
                # 成功抓取：闭合夹爪，回原点
                robot.close_gripper()
                time.sleep(1.0)
                robot.move_home_scripted()
            
            # 循环回到最外层，暂停并等待下一次回车

    except KeyboardInterrupt:
        print("\nStopping Program...")
    finally:
        robot.close()
        print("Program Exited.")

if __name__ == "__main__":
    main()