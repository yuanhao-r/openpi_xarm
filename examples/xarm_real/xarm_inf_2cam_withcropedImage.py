import time
import cv2
import numpy as np
import jax
from xarm.wrapper import XArmAPI

# OpenPI 依赖
import sys
import os
# 确保包含 openpi 文件夹的上一层目录 (src) 在路径中
# 获取当前脚本所在目录的上级目录（/home/openpi/examples）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接得到 openpi-client 的绝对路径（/home/openpi/packages/openpi-client）
openpi_client_path = os.path.join(current_dir, "../../packages/openpi-client/src")

# 确保包含 openpi 文件夹的上一层目录 (src) 在路径中
sys.path.append("/home/openpi/src")
# 新增：添加 openpi-client 所在目录到路径
sys.path.append(os.path.abspath(openpi_client_path))

# 现在可以正常导入
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
# CHECKPOINT_DIR = "/home/openpi/checkpoints/1222_night/10hz/8000"     #实验三
CHECKPOINT_DIR = "/home/openpi/checkpoints/exp9/94000"          #实验四
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
EXECUTE_STEPS = 1 # 每次推理执行多少步 (Chunk execution)

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
MIN_SAFE_Z = -69
# 减速因子：1.0 为原速，2.0 为 2倍慢，3.0 为 3倍慢。
# 建议从 2.0 或 3.0 开始尝试，直到你觉得速度合适。
SLOW_DOWN_FACTOR = 3.0  

# 插值频率 (Hz)：发送给机械臂指令的频率，越高越丝滑，建议 100Hz 或 50Hz
INTERPOLATION_FREQ = 100.0 

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
        # 手动构造全黑的 cam_high
        dummy_high = np.zeros((224, 224, 3), dtype=np.uint8) 
        obs["cam_high"] = dummy_high

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

    # def execute_action(self, action_rad):
    #     """
    #     执行单步动作 (输入是弧度)
    #     """
    #     target_joints_rad = action_rad[:6]
    #     target_gripper = action_rad[6]

    #     # 1. 安全限位
    #     safe_joints = []
    #     for i, angle in enumerate(target_joints_rad):
    #         low, high = JOINT_LIMITS[i]
    #         clipped = np.clip(angle, low, high)
    #         safe_joints.append(clipped)
        
    #     # -------------------------------------------------------
    #     # Z轴安全检查逻辑
    #     # -------------------------------------------------------
    #     # 1. 计算正运动学 (FK)，获取预测的 Cartesian 坐标
    #     # input_is_radian=True, return_is_radian=True
    #     # ret, pose = self.arm.get_forward_kinematics(angles=safe_joints, input_is_radian=True, return_is_radian=True)
        
    #     # if ret == 0:
    #     #     pred_x, pred_y, pred_z, pred_roll, pred_pitch, pred_yaw = pose
            
    #     #     # 2. 检查 Z 轴是否低于限制
    #     #     if pred_z < MIN_SAFE_Z:
    #     #         # print(f"[Safety] Z轴过低 ({pred_z:.1f} < {MIN_SAFE_Z})! 正在修正...")
                
    #     #         # 3. 构造修正后的目标位姿 (Clamped Pose)
    #     #         # 保持 X, Y, Roll, Pitch, Yaw 不变，强行把 Z 拉回到安全高度
    #     #         safe_pose = [pred_x, pred_y, MIN_SAFE_Z, pred_roll, pred_pitch, pred_yaw]
                
    #     #         # 4. 逆运动学 (IK) 解算回关节角度
    #     #         ret_ik, ik_joints = self.arm.get_inverse_kinematics(safe_pose, input_is_radian=True, return_is_radian=True)
                
    #     #         if ret_ik == 0:
    #     #             # 如果 IK 解算成功，用修正后的关节角度覆盖
    #     #             safe_joints = ik_joints
    #     #         else:
    #     #             print("[Safety Error] IK解算失败，无法修正Z轴。跳过此步运动。")
    #     #             return # 直接跳过这次指令，防止碰撞
    #     # else:
    #     #     # 如果算不出 FK (通常不会发生)，为安全起见可以选择跳过
    #     #     pass 
    #     # -------------------------------------------------------

    #     full_joints = np.append(safe_joints, 0.0)

    #     # 2. 发送伺服指令 (弧度)
    #     ret = self.arm.set_servo_angle_j(angles=full_joints, is_radian=True)
    #     if ret != 0:
    #         print(f"[Error] Servo failed: {ret}")

    #     # 3. 夹爪逻辑
    #     if target_gripper > 0.8:
    #         self.close_gripper()
    #     elif target_gripper < 0.2:
    #         self.open_gripper()
    def execute_action(self, action_rad):
        """
        执行单步动作，包含 Z轴限位 + 平滑减速插值
        """
        target_joints_rad = action_rad[:6]
        target_gripper = action_rad[6]

        # 1. 关节角度物理限位
        safe_joints = []
        for i, angle in enumerate(target_joints_rad):
            low, high = JOINT_LIMITS[i]
            clipped = np.clip(angle, low, high)
            safe_joints.append(clipped)

        # -------------------------------------------------------
        # Z轴安全检查逻辑 (正运动学 FK + 逆运动学 IK)
        # -------------------------------------------------------
        # 计算目标姿态
        ret, pose = self.arm.get_forward_kinematics(angles=safe_joints, input_is_radian=True, return_is_radian=True)
        
        if ret == 0:
            pred_x, pred_y, pred_z, pred_roll, pred_pitch, pred_yaw = pose
            
            # 如果预测高度低于安全值，强行修正
            if pred_z < MIN_SAFE_Z:
                # print(f"[Safety] Limit Z: {pred_z:.1f} -> {MIN_SAFE_Z}")
                # 保持其他维度不变，只修改 Z
                safe_pose = [pred_x, pred_y, MIN_SAFE_Z, pred_roll, pred_pitch, pred_yaw]
                
                # IK 解算回关节角度
                ret_ik, ik_joints = self.arm.get_inverse_kinematics(safe_pose, input_is_radian=True, return_is_radian=True)
                
                if ret_ik == 0:
                    safe_joints = list(ik_joints) # 使用修正后的关节角
                else:
                    print("[Safety] IK Failed, skipping step.")
                    return # 解算失败则跳过
        # -------------------------------------------------------

        # -------------------------------------------------------
        # 【核心修改】平滑减速插值 (Time Dilation)
        # -------------------------------------------------------
        # 获取当前机械臂的关节角度作为起点
        code, current_joints_raw = self.arm.get_servo_angle(is_radian=True)
        if code != 0:
            print("[Error] Failed to get current joints, skip interpolation.")
            return

        current_joints = np.array(current_joints_raw[:6])
        target_joints = np.array(safe_joints[:6])

        # 计算这一步应该花多少时间
        # 原本是 1/10Hz = 0.1秒。现在乘以减速因子。
        # 例如 SLOW_DOWN_FACTOR = 3.0，则 duration = 0.3秒
        duration = (1.0 / CONTROL_FREQ) * SLOW_DOWN_FACTOR
        
        # 计算需要插值多少步
        # 例如 0.3秒 * 100Hz = 30步
        steps = int(duration * INTERPOLATION_FREQ)
        
        if steps < 1: steps = 1
        
        # 循环执行微小的插值动作
        for i in range(1, steps + 1):
            alpha = i / steps # 进度 0.0 ~ 1.0
            
            # 线性插值公式: 当前 + (目标-当前)*进度
            interp_joints = current_joints + (target_joints - current_joints) * alpha
            
            # 补上第7维(无效位，SDK要求7维)
            full_joints_interp = np.append(interp_joints, 0.0)
            
            # 发送给机械臂
            self.arm.set_servo_angle_j(angles=full_joints_interp, is_radian=True)
            
            # 睡眠控制插值频率 (例如 0.01秒)
            time.sleep(1.0 / INTERPOLATION_FREQ)

        # -------------------------------------------------------
        # 3. 夹爪逻辑 (动作执行完后再操作夹爪)
        # -------------------------------------------------------
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