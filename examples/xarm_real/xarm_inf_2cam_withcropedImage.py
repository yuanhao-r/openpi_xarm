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
CONFIG_NAME = "pi05_xarm_1212_night" 
# 请确认这是你最新的、用【弧度】数据训练的 checkpoint
# CHECKPOINT_DIR = "/home/hil-serl/openpi_test/openpi/checkpoints/1207night/19999" 
CHECKPOINT_DIR = "/home/hil-serl/openpi_test/openpi/checkpoints/pi05_xarm_1222_weekend_2cam4090/142000" 

CAMERAS = {
    # "cam_high": 4,
    "cam_left_wrist": 0,
    "cam_right_wrist": 2
}
CROP_CONFIGS = {
    # "cam_high": (61, 1, 434, 479),       # x, y, w, h
    "cam_left_wrist": (118, 60, 357, 420),
    "cam_right_wrist": (136, 57, 349, 412)
}

CONTROL_FREQ = 10 # 10Hz
EXECUTE_STEPS = 4 # 每次推理执行多少步 (Chunk execution)

# 关节限位 (Rad) - 用于安全限幅
JOINT_LIMITS = [
    (-6.2, 6.2),     # J1
    (-2.0, 2.0),     # J2
    (-2.9, 2.9),     # J3 
    (-3.1, 3.1),     # J4
    (-1.6, 1.8),     # J5
    (-6.2, 6.2)      # J6
]

# 【脚本控制】Home 点坐标 (Cartesian: x, y, z, roll, pitch, yaw)
# 这里的 Z 给高一点 (200)，防止回 Home 时撞到东西
# HOME_POS = [616.942688, -181.296143, 56.682617, -3.125365, 0.002198, 0.98222]
HOME_POS = [539.120605, 17.047951, 100-59.568863, 3.12897, 0.012689, -1.01436]
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
        
        # 初始模式：Mode 0 (位置规划模式)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        
        # 初始动作：张开夹爪
        self.open_gripper()
        time.sleep(1.0) 

    def close_gripper(self):
        # 强制发送指令，确保夹紧
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 1.0
        print(">>> [Hardware] Gripper Closed.")

    def open_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 0.0
        print(">>> [Hardware] Gripper Opened.")
 
 
    def get_observation(self) -> dict:
        """获取观测值 (直接使用弧度)"""
        obs = {}
        for name, cap in self.caps.items():
            ret, frame = cap.read()
            if not ret: 
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # ========== 先裁剪 ==========
            if name in CROP_CONFIGS:
                x, y, w, h = CROP_CONFIGS[name]
                # OpenCV裁剪：frame[y:y+h, x:x+w]
                frame = frame[y:y+h, x:x+w]
            # =============================
            # frame = image_tools.resize_with_pad(frame, 224, 224)
            obs[name] = image_tools.convert_to_uint8(frame)
            print(f"[{name}] 图片尺寸: {frame.shape}")
    
        # 1. 获取物理角度 (弧度)
        code, joints_rad = self.arm.get_servo_angle(is_radian=True)
        if code != 0: 
            print(f"[Warn] Get joint error: {code}")
            joints_rad = [0.0] * 7
        
        # 2. 【直接使用弧度】不再乘以 RAD_2_DEG
        state_vec = np.array(joints_rad[:6], dtype=np.float32)
        
        # 3. 拼装夹爪
        state_vec_final = np.append(state_vec, self.current_gripper_state)
        
        obs["state"] = state_vec_final
        return obs

    def move_to_start(self, target_action_rad):
        """
        使用 Mode 0 安全移动到起始点 (输入已经是弧度)
        """
        print(">>> Moving to start position (Mode 0)...")
        
        target_joints = target_action_rad[:6]
        
        # 安全限位
        safe_joints = []
        for i, angle in enumerate(target_joints):
            low, high = JOINT_LIMITS[i]
            safe_joints.append(np.clip(angle, low, high))
            
        self.arm.set_mode(0)
        self.arm.set_state(0)
        # speed 0.35 rad/s 约等于 20 deg/s
        self.arm.set_servo_angle(angle=safe_joints, speed=0.35, is_radian=True, wait=True)
        
        print(">>> Reached start. Switching to Mode 1 (Servo)...")
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.5)

    def execute_action(self, action_rad):
        """
        执行单步动作 (输入是弧度)
        """
        target_joints_rad = action_rad[:6]
        target_gripper = action_rad[6]

        # 1. 安全限位
        safe_joints = []
        for i, angle in enumerate(target_joints_rad):
            low, high = JOINT_LIMITS[i]
            clipped = np.clip(angle, low, high)
            safe_joints.append(clipped)
        
        full_joints = np.append(safe_joints, 0.0)

        # 2. 发送伺服指令 (弧度)
        ret = self.arm.set_servo_angle_j(angles=full_joints, is_radian=True)
        if ret != 0:
            print(f"[Error] Servo failed: {ret}")

        # 3. 夹爪逻辑
        if target_gripper > 0.8:
            self.close_gripper()
        elif target_gripper < 0.2:
            self.open_gripper()
    
    def move_home_scripted(self):
        """
        【脚本控制】切换回规划模式并移动到 Home 点
        """
        print("\n>>> [System] Switching to Script Mode (Mode 0) to return HOME...")
        self.arm.set_mode(0)
        self.arm.set_state(0)
        
        # 使用笛卡尔坐标移动到 Home
        self.arm.set_position(
            x=HOME_POS[0], y=HOME_POS[1], z=HOME_POS[2],
            roll=HOME_POS[3], pitch=HOME_POS[4], yaw=HOME_POS[5],
            speed=100, wait=True, is_radian=True
        )
        print(">>> [System] Returned to Home.")

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
    
    # prompt = "pickup"
    # prompt = "pickupA"
    # prompt = "pick up the industrial components A"
    prompt = "pick up the industrial components"

    # C. 【预推理】 获取第一个点并复位
    print("\n[Phase 1] Pre-inferencing to alignment...")
    raw_obs = robot.get_observation()
    example = {
        # "cam_high": raw_obs["cam_high"],
        "cam_left_wrist": raw_obs["cam_left_wrist"],
        "cam_right_wrist": raw_obs["cam_right_wrist"],
        "state": raw_obs["state"],
        "prompt": prompt
    }
    
    result = policy.infer(example)
    start_action_rad = np.array(result["actions"])[0] # 模型输出现在是弧度
    
    # 移动到起始点
    robot.move_to_start(start_action_rad)

    # D. 【主循环】 AI 控制
    print("\n[Phase 2] Start AI Servo Control (Ctrl+C to stop)...")
    try:
        while True:
            # 1. 观测
            raw_obs = robot.get_observation()
            example = {
                # "cam_high": raw_obs["cam_high"],
                "cam_left_wrist": raw_obs["cam_left_wrist"],
                "cam_right_wrist": raw_obs["cam_right_wrist"],
                "state": raw_obs["state"],
                "prompt": prompt
            }

            # 2. 推理
            result = policy.infer(example)
            action_chunk = np.array(result["actions"]) 

            # =========================================================
            # 【新增】终止条件检测
            # 检测前 5 步的夹爪状态。如果大部分是闭合 (>0.8)，则终止模型。
            # =========================================================
            check_steps = 1
            gripper_preds = action_chunk[:check_steps, 6] # 取前5步的第7维(夹爪)
            
            # 计算前5步夹爪状态的均值
            # gripper_mean = np.mean(gripper_preds)
            print("前5步的第7维(夹爪):", gripper_preds)
             # np.any() 检查数组中是否存在任意元素满足条件
            if np.any(gripper_preds > 0.8):
                print("\n" + "!"*60)
                # 找出是第几步触发的，方便调试看数据
                first_trigger_idx = np.where(gripper_preds > 0.8)[0][0]
                trigger_val = gripper_preds[first_trigger_idx]
                
                print(f"检测到抓取意图 (在第 {first_trigger_idx} 步数值为 {trigger_val:.4f})")
                print(f"前{check_steps}步夹爪数值: {gripper_preds}")
                print(">>> 终止模型控制，转入脚本控制序列...")
                print("!"*60 + "\n")
                
                # 1. 确保夹紧
                robot.close_gripper()
                time.sleep(2.0)
                
                # 2. 执行脚本回 Home
                robot.move_home_scripted()
                
                # 3. 退出循环，结束程序
                break
            # =========================================================

            # 3. 正常执行 Chunk
            steps_to_run = min(EXECUTE_STEPS, len(action_chunk))
            
            for i in range(steps_to_run):
                step_start = time.time()
                
                # 执行动作 (传入弧度)
                robot.execute_action(action_chunk[i])
                
                # 频率控制
                elapsed = time.time() - step_start
                sleep_time = (1.0 / CONTROL_FREQ) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.close()
        print("Program Finished.")

if __name__ == "__main__":
    main()