import time
import cv2
import numpy as np
import jax
from xarm.wrapper import XArmAPI

# OpenPI 依赖
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools 

# -----------------------------------------------------------------------------
# 1. 配置区域
# -----------------------------------------------------------------------------
ROBOT_IP = "192.168.1.232"
CONFIG_NAME = "pi05_xarm" 
# 请修改为你的实际 checkpoint 路径
CHECKPOINT_DIR = "/home/hil-serl/openpi_test/openpi/checkpoints/19999" 

CAMERAS = {
    "cam_high": 4,
    "cam_left_wrist": 0,
    "cam_right_wrist": 2
}

CONTROL_FREQ = 10 # 10Hz
EXECUTE_STEPS = 4 # 每次推理执行4步

# === [关键开关] ===
# True  = 模型训练使用的是相对值 (observation.state_delta / action_delta)
# False = 模型训练使用的是绝对值 (observation.state / action)
USE_RELATIVE_MODEL = False  

# 相对值放大倍数 (必须与 convert 脚本一致)
SCALE_FACTOR = 1000.0 

# 关节限位 (Rad)
JOINT_LIMITS = [
    (-6.2, 6.2),     # J1
    (-2.0, 2.0),     # J2
    (-2.9, 2.9),     # J3 
    (-3.1, 3.1),     # J4
    (-1.6, 1.8),     # J5
    (-6.2, 6.2)      # J6
]

# 安全限幅：绝对位置模式下，单步最大允许突变弧度
# 防止模型突然输出一个很远的点
MAX_JUMP_RAD = 0.5 

# -----------------------------------------------------------------------------
# 2. 硬件封装类
# -----------------------------------------------------------------------------
class XArmHardware:
    def __init__(self, ip, camera_indices):
        print(f"Connecting to xArm at {ip}...")
        self.arm = XArmAPI(ip)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_tgpio_modbus_baudrate(baud=115200)
        
        # 初始化相机
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
        
        # 初始模式：Mode 0 (位置规划模式) 用于复位
        self.arm.set_mode(0)
        self.arm.set_state(0)
        
        # 初始动作：张开夹爪
        self.open_gripper()
        time.sleep(1.0) 

    def close_gripper(self):
        if self.current_gripper_state > 0.9: return
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 1.0
        print(">>> Gripper Closing")

    def open_gripper(self):
        if self.current_gripper_state < 0.1: return
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 0.0
        print(">>> Gripper Opening")

    def get_observation(self) -> dict:
        """获取观测值"""
        obs = {}
        for name, cap in self.caps.items():
            ret, frame = cap.read()
            if not ret: 
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = image_tools.resize_with_pad(frame, 224, 224)
            obs[name] = image_tools.convert_to_uint8(frame)

        # 1. 获取物理角度 (弧度)
        code, joints_rad = self.arm.get_servo_angle(is_radian=True)
        if code != 0: 
            print(f"[Warn] Get joint error: {code}")
            joints_rad = [0.0] * 7
        
        # 2. 根据模式决定 State 的格式
        if USE_RELATIVE_MODEL:
            # 相对值模型通常需要上一帧的动作作为输入，或者仍然输入绝对位置
            # OpenPi 默认是输入绝对位置 (Proprioception)
            # 所以这里不管是练相对还是绝对，State 都给绝对弧度通常没问题
            # 除非你特意训练了 State 也是 Delta 的模型 (通常不推荐)
            state_vec = np.array(joints_rad[:6], dtype=np.float32)
        else:
            # 绝对值模式：直接用弧度
            state_vec = np.array(joints_rad[:6], dtype=np.float32)
        
        # 3. 拼装夹爪
        state_vec_final = np.append(state_vec, self.current_gripper_state)
        
        obs["state"] = state_vec_final
        return obs

    def move_to_start(self, start_action):
        """
        移动到起始点 (Mode 0)
        start_action: 模型的第一个输出
        """
        print(">>> Moving to start position (Mode 0)...")
        
        if USE_RELATIVE_MODEL:
            # 相对值模式下，第一个动作是 Delta，没法直接知道绝对位置在哪
            # 只能假设当前就在起始点附近，或者忽略这一步
            print("[Info] Relative Mode: Skipping move_to_start (Delta is unknown)")
            target_rad = None
        else:
            # 绝对值模式：直接取前6维弧度
            target_rad = start_action[:6]

        if target_rad is not None:
            # 安全限位
            safe_joints = []
            for i, angle in enumerate(target_rad):
                low, high = JOINT_LIMITS[i]
                safe_joints.append(np.clip(angle, low, high))
                
            self.arm.set_mode(0)
            self.arm.set_state(0)
            self.arm.set_servo_angle(angle=safe_joints, speed=0.35, is_radian=True, wait=True)
        
        print(">>> Switching to Mode 1 (Servo)...")
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.5)

    def execute_action(self, action_pred):
        """
        执行单步动作
        """
        target_joints_raw = action_pred[:6]
        target_gripper = action_pred[6]

        # 打印夹爪状态 (调试用)
        # print(f"Gripper Pred: {target_gripper:.4f}")

        # === 核心分支逻辑 ===
        if USE_RELATIVE_MODEL:
            # --- 相对值模式 ---
            # 1. 获取当前真实角度
            code, curr_joints = self.arm.get_servo_angle(is_radian=True)
            if code != 0: return
            curr_joints = np.array(curr_joints[:6])

            # 2. 反缩放 (模型输出是 1000x 的 Delta)
            delta_rad = target_joints_raw / SCALE_FACTOR
            
            # 3. 累加得到目标绝对值
            target_rad = curr_joints + delta_rad
            
        else:
            # --- 绝对值模式 ---
            # 模型直接输出目标弧度
            target_rad = target_joints_raw
            
            # (可选) 防止跳变保护
            # code, curr_joints = self.arm.get_servo_angle(is_radian=True)
            # if code == 0:
            #     diff = np.abs(target_rad - curr_joints[:6])
            #     if np.max(diff) > MAX_JUMP_RAD:
            #         print(f"[Danger] Action jump too large! Max diff: {np.max(diff):.2f}")
            #         return # 跳过执行

        # --- 统一发送逻辑 ---
        # 安全限位
        safe_joints = []
        for i, angle in enumerate(target_rad):
            low, high = JOINT_LIMITS[i]
            clipped = np.clip(angle, low, high)
            safe_joints.append(clipped)
        
        # 补全7维 (xArm SDK要求)
        full_joints = np.append(safe_joints, 0.0) 

        # 发送伺服指令
        ret = self.arm.set_servo_angle_j(angles=full_joints, is_radian=True)
        if ret != 0:
            # 偶尔出现 code=1 (Velocity Limit) 是正常的，说明动太快了被限速
            # 如果出现 code=60 就要小心了
            pass 

        # 夹爪逻辑 (迟滞阈值)
        # 你的数据只有 0.0 和 1.0，所以这里用 0.5 作为分界线是合理的
        # 如果模型不自信(例如输出 0.3)，可以用更激进的阈值比如 0.2
        if target_gripper > 0.5:
            self.close_gripper()
        else:
            self.open_gripper()

    def close(self):
        self.arm.set_mode(0)
        self.arm.set_state(0)
        for cap in self.caps.values():
            cap.release()
        self.arm.disconnect()

# -----------------------------------------------------------------------------
# 3. 主程序逻辑
# -----------------------------------------------------------------------------
def main():
    # A. 加载模型
    print(f"Loading Config: {CONFIG_NAME}...")
    config = _config.get_config(CONFIG_NAME)
    print(f"Loading Checkpoint from: {CHECKPOINT_DIR}...")
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    print("Model Loaded.")

    # B. 初始化硬件
    robot = XArmHardware(ROBOT_IP, CAMERAS)
    prompt = input("请输入指令 (回车默认 'pick object'): ") or "pick object"

    # C. 【预推理】 获取第一个点并复位
    # 如果是绝对值模式，可以让机械臂先走到起始点，防止一开始就狂奔
    print("\n[Phase 1] Pre-inferencing...")
    raw_obs = robot.get_observation()
    example = {
        "cam_high": raw_obs["cam_high"],
        "cam_left_wrist": raw_obs["cam_left_wrist"],
        "cam_right_wrist": raw_obs["cam_right_wrist"],
        "state": raw_obs["state"], # 始终输入绝对值 (根据OpenPi默认配置)
        "prompt": prompt
    }
    
    result = policy.infer(example)
    start_action = np.array(result["actions"])[0] 
    
    robot.move_to_start(start_action)

    # D. 【主循环】 AI 控制
    print("\n[Phase 2] Start AI Servo Control (Ctrl+C to stop)...")
    try:
        while True:
            step_start = time.time()
            
            # 1. 观测
            raw_obs = robot.get_observation()
            example = {
                "cam_high": raw_obs["cam_high"],
                "cam_left_wrist": raw_obs["cam_left_wrist"],
                "cam_right_wrist": raw_obs["cam_right_wrist"],
                "state": raw_obs["state"],
                "prompt": prompt
            }

            # 2. 推理
            result = policy.infer(example)
            action_chunk = np.array(result["actions"]) 

            # 3. 执行 Chunk
            steps_to_run = min(EXECUTE_STEPS, len(action_chunk))
            
            for i in range(steps_to_run):
                loop_start = time.time()
                
                # 执行动作
                robot.execute_action(action_chunk[i])
                
                # 频率控制
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / CONTROL_FREQ) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.close()

if __name__ == "__main__":
    main()