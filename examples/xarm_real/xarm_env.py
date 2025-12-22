import time
import cv2
import numpy as np
from xarm.wrapper import XArmAPI

# OpenPI Client 依赖
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# -----------------------------------------------------------------------------
# 配置参数
# -----------------------------------------------------------------------------
# xArm IP
ROBOT_IP = "192.168.1.232"

# Policy Server 地址 (运行 serve_policy.py 的电脑 IP)
SERVER_HOST = "0.0.0.0" # 如果在同一台机器，用 localhost
SERVER_PORT = 8000

# 摄像头配置 (对应你的三个相机)
CAMERAS = {
    "cam_high": 4,
    "cam_left_wrist": 0,
    "cam_right_wrist": 2
}

# 控制参数
CONTROL_FREQ = 15  # 控制循环频率 (Hz)
EXECUTE_STEPS = 8  # 每次推理后，执行 Action Chunk 中的前多少步 (Receding Horizon)

# -----------------------------------------------------------------------------
# 硬件控制类 (负责与 xArm 和相机交互)
# -----------------------------------------------------------------------------
class XArmRobot:
    def __init__(self, ip, camera_indices):
        print(f"Connecting to xArm at {ip}...")
        self.arm = XArmAPI(ip)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_tgpio_modbus_baudrate(baud=115200)
        
        # 初始复位
        self._reset_home()
        
        # 切换到 Mode 1 (关节伺服)，适合连续轨迹控制
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.5)
        print("xArm Ready (Mode 1).")

        # 初始化相机
        self.caps = {}
        for name, idx in camera_indices.items():
            cap = cv2.VideoCapture(idx)
            # 固定分辨率，防止拼接错误
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.caps[name] = cap
            else:
                print(f"[Error] Failed to open camera {name} (Idx {idx})")
        
        self.current_gripper_state = 0.0 # 0=Open, 1=Closed
        self.open_gripper()

    def _reset_home(self):
        self.arm.set_mode(0)
        self.arm.set_state(0)
        # 你的初始位姿
        home_joints = [0.0, -0.26, 0.0, 0.52, 0.0, 1.57, 0.0]
        self.arm.set_servo_angle(angle=home_joints, speed=0.35, is_radian=True, wait=True)

    def close_gripper(self):
        if self.current_gripper_state == 1.0: return
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 1.0

    def open_gripper(self):
        if self.current_gripper_state == 0.0: return
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 0.0

    def get_observation(self) -> dict:
        """
        获取原始观测数据 (未缩放的图像 + 状态)
        """
        obs = {}
        # 1. 获取图像
        for name, cap in self.caps.items():
            ret, frame = cap.read()
            if not ret: 
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # 转 RGB
            obs[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. 获取状态 (关节角)
        code, joints = self.arm.get_servo_angle(is_radian=True)
        if code != 0: joints = [0.0] * 7
        
        # 拼装 State: 6个关节角 + 1个夹爪状态
        # 必须是 (7,) 维数组
        obs["state"] = np.array(joints[:6] + [self.current_gripper_state], dtype=np.float32)
        
        return obs

    def execute_action(self, action):
        """执行单步动作"""
        # action shape: (7,) -> [j1...j6, gripper]
        target_joints = action[:6]
        target_gripper = action[6]

        # 关节运动 (补全第7轴为0)
        full_joints = np.append(target_joints, 0.0)
        self.arm.set_servo_angle_j(angles=full_joints, is_radian=True)

        # 夹爪运动
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
# 主程序
# -----------------------------------------------------------------------------
def main():
    # 1. 初始化 Client (连接服务器)
    print(f"Connecting to Policy Server at {SERVER_HOST}:{SERVER_PORT}...")
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=SERVER_HOST, 
            port=SERVER_PORT
        )
        print("Connected.")
        print(f"Metadata: {client.get_server_metadata()}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # 2. 初始化硬件
    robot = XArmRobot(ROBOT_IP, CAMERAS)

    # 3. 输入指令
    task_instruction = input("请输入任务指令 (例如 'pick up the red block'): ")
    if not task_instruction: task_instruction = "do something"

    print("开始闭环控制... 按 Ctrl+C 停止")

    try:
        while True:
            # --- 步骤 A: 构建 Observation ---
            # 获取硬件原始数据
            raw_data = robot.get_observation()
            
            # 使用 image_tools 处理数据 (参考 README)
            # 这里的 Key 必须和你训练时的 RepackTransform 里的 Key 一致
            observation = {
                "cam_high": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(raw_data["cam_high"], 224, 224)
                ),
                "cam_left_wrist": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(raw_data["cam_left_wrist"], 224, 224)
                ),
                "cam_right_wrist": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(raw_data["cam_right_wrist"], 224, 224)
                ),
                # 状态不需要归一化，服务端会处理
                "state": raw_data["state"], 
                "prompt": task_instruction,
            }

            # --- 步骤 B: 发送请求获取 Action Chunk ---
            start_infer = time.time()
            # client.infer 返回的是 {"actions": array, ...}
            # actions shape 通常是 (Chunk_Size, 7)
            response = client.infer(observation)
            action_chunk = response["actions"]
            # print(f"Infer time: {(time.time()-start_infer)*1000:.1f}ms")

            # --- 步骤 C: 执行 Action Chunk (Open-Loop Execution) ---
            # 我们只执行前 EXECUTE_STEPS 步，然后重新循环去获取最新图像
            steps_to_run = min(EXECUTE_STEPS, len(action_chunk))
            
            for i in range(steps_to_run):
                loop_start = time.time()
                
                # 取出当前步动作
                action = action_chunk[i]
                
                # 发送给 xArm
                robot.execute_action(action)
                
                # 频率控制 (保持稳定的 15Hz)
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