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
# CHECKPOINT_DIR = "/home/hil-serl/openpi_test/openpi/checkpoints/my_experiment/1000" 
CHECKPOINT_DIR = "/home/hil-serl/openpi_test/openpi/checkpoints/19999" 

CAMERAS = {
    "cam_high": 4,
    "cam_left_wrist": 0,
    "cam_right_wrist": 2
}

CONTROL_FREQ = 10 # 10Hz
EXECUTE_STEPS = 4 # 每次推理执行4步

# 单位转换常量
RAD_2_DEG = 180.0 / np.pi
DEG_2_RAD = np.pi / 180.0

# 关节限位 (Rad)
JOINT_LIMITS = [
    (-6.2, 6.2),     # J1
    (-2.0, 2.0),     # J2
    (-2.9, 2.9),     # J3 
    (-3.1, 3.1),     # J4
    (-1.6, 1.8),     # J5
    (-6.2, 6.2)      # J6
]

# 安全钳位：单步最大允许转动弧度 (防止模型抽风)
# 0.1 rad/0.1s = 1 rad/s (约57度/秒)，这是一个比较激进但安全的上限
MAX_STEP_DELTA = 0.1 

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
        time.sleep(1.0) # 等待夹爪动作

    def close_gripper(self):
        if self.current_gripper_state > 0.9: return
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 1.0

    def open_gripper(self):
        if self.current_gripper_state < 0.1: return
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 0.0

    def get_observation(self) -> dict:
        """获取观测值，并将关节角度转换为【度】喂给模型"""
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
        
        # 2. 【关键转换】 弧度 -> 度
        # 模型是用度训练的，所以这里必须转成度
        state_vec_deg = np.array(joints_rad[:6], dtype=np.float32) * RAD_2_DEG
        
        # 3. 拼装夹爪 (夹爪通常是 0-1，不需要转换)
        state_vec_final = np.append(state_vec_deg, self.current_gripper_state)
        
        obs["state"] = state_vec_final
        return obs

    def move_to_start(self, target_action_deg):
        """
        使用 Mode 0 安全移动到起始点
        输入: target_action_deg (包含单位为【度】的关节角)
        """
        print(">>> Moving to start position (Mode 0)...")
        
        # 1. 【关键转换】 度 -> 弧度 (因为 API 需要弧度)
        target_joints_rad = target_action_deg[:6] * DEG_2_RAD
        
        # 2. 安全限位
        safe_joints = []
        for i, angle in enumerate(target_joints_rad):
            low, high = JOINT_LIMITS[i]
            safe_joints.append(np.clip(angle, low, high))
            
        # 3. 执行移动 (慢速)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_servo_angle(angle=safe_joints, speed=0.35, is_radian=True, wait=True)
        
        print(">>> Reached start. Switching to Mode 1 (Servo)...")
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.5)

    def execute_action(self, action_deg):
        """
        执行单步动作
        输入: action_deg (模型输出的，单位为【度】)
        """
        target_joints_deg = action_deg[:6]
        target_gripper = action_deg[6]
        print(f"\n★★★★★ [夹爪指令] ★★★★★")
        print(f"【target_gripper 数值】: {target_gripper:.4f}")
        print(f"【当前夹爪状态】: {'闭合' if self.current_gripper_state > 0.5 else '张开'}")
        print(f"★★★★★ [/夹爪指令] ★★★★★\n")
        # 1. 【关键转换】 度 -> 弧度
        target_joints_rad = target_joints_deg * DEG_2_RAD

        # 2. 安全限位 (Clamping Limits)
        safe_joints = []
        for i, angle in enumerate(target_joints_rad):
            low, high = JOINT_LIMITS[i]
            clipped = np.clip(angle, low, high)
            safe_joints.append(clipped)
        
        full_joints = np.append(safe_joints, 0.0) # 补齐长度

        # 3. 发送伺服指令
        # set_servo_angle_j 是非阻塞的，适合高频控制
        ret = self.arm.set_servo_angle_j(angles=full_joints, is_radian=True)
        if ret != 0:
            # Error 60 可能会在这里偶尔出现，如果出现，通常是因为 target 跳变太大
            print(f"[Error] Servo failed: {ret}")

        # 4. 夹爪逻辑
        if target_gripper > 0.8:
            self.close_gripper()
        elif target_gripper < 0.2:
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
    print("\n[Phase 1] Pre-inferencing to alignment...")
    raw_obs = robot.get_observation() # 这里已经是【度】了
    example = {
        "cam_high": raw_obs["cam_high"],
        "cam_left_wrist": raw_obs["cam_left_wrist"],
        "cam_right_wrist": raw_obs["cam_right_wrist"],
        "state": raw_obs["state"],
        "prompt": prompt
    }
    
    result = policy.infer(example)
    start_action_deg = np.array(result["actions"])[0] # 模型输出是【度】
    
    # 移动到起始点 (内部会转成弧度发给机械臂)
    robot.move_to_start(start_action_deg)

    # D. 【主循环】 AI 控制
    print("\n[Phase 2] Start AI Servo Control (Ctrl+C to stop)...")
    try:
        while True:
            loop_start = time.time()
            
            # 1. 观测 (API弧度 -> 转度)
            raw_obs = robot.get_observation()
            example = {
                "cam_high": raw_obs["cam_high"],
                "cam_left_wrist": raw_obs["cam_left_wrist"],
                "cam_right_wrist": raw_obs["cam_right_wrist"],
                "state": raw_obs["state"],
                "prompt": prompt
            }

            # 2. 推理 (输入度 -> 输出度)
            result = policy.infer(example)
            action_chunk_deg = np.array(result["actions"]) 

            # 3. 执行 Chunk (输出度 -> 转弧度 -> 发送)
            steps_to_run = min(EXECUTE_STEPS, len(action_chunk_deg))
            
            for i in range(steps_to_run):
                step_start = time.time()
                
                # 执行动作
                robot.execute_action(action_chunk_deg[i])
                
                # 频率控制
                elapsed = time.time() - step_start
                sleep_time = (1.0 / CONTROL_FREQ) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # 这里的总体延迟补偿 (可选)
            # total_elapsed = time.time() - loop_start
            # print(f"Loop FPS: {1.0/total_elapsed:.1f}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.close()

if __name__ == "__main__":
    main()