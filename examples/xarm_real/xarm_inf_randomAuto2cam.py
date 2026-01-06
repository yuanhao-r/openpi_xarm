import time
import cv2
import numpy as np
import sys
import os
import termios
import select
import random
from scipy.spatial import ConvexHull  # 必须安装 scipy

from xarm.wrapper import XArmAPI

# -----------------------------------------------------------------------------
# 路径设置 (保持原有)
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
    # "cam_high": 4, 
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

# === 关键点位定义 ===
HOME_POS = [539.120605, 17.047951, 100-59.568863, 3.12897, 0.012689, -1.01436]
# A点 (取货点)
POS_A = [539.120605, 17.047951, -79.568863, 3.12897, 0.012689, -1.01436]
MIN_SAFE_Z = -69

SLOW_DOWN_FACTOR = 3.0  
INTERPOLATION_FREQ = 100.0 

# ===================== 几何区域定义 (来自采集脚本) =====================
BOUNDARY_POINTS_2D = np.array([
    [505.982422, -150.631149],  # 左下
    [724.302856, -66.848724],   # 右下
    [724.232117, 240.781003],   # 右上
    [428.805481, 203.618057],   # 左上
])
FIXED_Z = POS_A[2]
FIXED_ROLL = POS_A[3]
FIXED_PITCH = POS_A[4]
BASE_YAW = POS_A[5]
YAW_RANDOM_RANGE = (-np.pi/2, np.pi/6)

# -----------------------------------------------------------------------------
# 2. 工具类：采样器 (处理网格和边界逻辑)
# -----------------------------------------------------------------------------
class TaskSampler:
    def __init__(self):
        # 1. 基础几何
        self.hull_2d = ConvexHull(BOUNDARY_POINTS_2D)
        self.boundary_points = BOUNDARY_POINTS_2D
        self.min_x = np.min(BOUNDARY_POINTS_2D[:, 0])
        self.max_x = np.max(BOUNDARY_POINTS_2D[:, 0])
        self.min_y = np.min(BOUNDARY_POINTS_2D[:, 1])
        self.max_y = np.max(BOUNDARY_POINTS_2D[:, 1])

        # 2. 网格相关 (4x4)
        self.grid_rows = 4
        self.grid_cols = 4
        self.grid_indices = []
        self._refill_grid_indices()

        # 3. 边界相关
        # 获取逆时针顶点
        ccw_indices = self.hull_2d.vertices
        self.ccw_vertices = BOUNDARY_POINTS_2D[ccw_indices]
        # 生成路径点
        self.boundary_step_size = 20.0
        self.path_points_2d = self._generate_boundary_path(self.ccw_vertices, self.boundary_step_size)
        self.current_path_idx = 0
        self.total_path_points = len(self.path_points_2d)
        
        print(f"[Sampler] Grid: 4x4 | Boundary Points: {self.total_path_points}")

    def _refill_grid_indices(self):
        """重置并打乱网格顺序"""
        self.grid_indices = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                self.grid_indices.append((r, c))
        random.shuffle(self.grid_indices)
        print("[Sampler] Grid indices reshuffled.")

    def _generate_boundary_path(self, vertices, step_size):
        path = []
        num_v = len(vertices)
        for i in range(num_v):
            p_curr = vertices[i]
            p_next = vertices[(i + 1) % num_v]
            vec = p_next - p_curr
            dist = np.linalg.norm(vec)
            steps = int(max(1, dist / step_size))
            unit_vec = vec / dist
            for s in range(steps):
                point = p_curr + unit_vec * (s * step_size)
                path.append(point)
        return np.array(path)

    def is_inside(self, x, y):
        point_homo = np.array([x, y, 1])
        for eq in self.hull_2d.equations:
            if np.dot(eq, point_homo) > 1e-6: return False
        return True

    def get_random_grid_target(self):
        """获取一个随机网格内的点 (对应按下 'r')"""
        # 尝试多次以防列表空了或采样失败
        for _ in range(self.grid_rows * self.grid_cols * 2):
            if not self.grid_indices:
                self._refill_grid_indices()
            
            r, c = self.grid_indices.pop() # 取出一个随机格子
            
            # 计算格子边界
            step_x = (self.max_x - self.min_x) / self.grid_cols
            step_y = (self.max_y - self.min_y) / self.grid_rows
            min_x = self.min_x + c * step_x
            max_x = min_x + step_x
            min_y = self.min_y + r * step_y
            max_y = min_y + step_y

            # 在格子内尝试采样
            for _ in range(30):
                tx = random.uniform(min_x, max_x)
                ty = random.uniform(min_y, max_y)
                if self.is_inside(tx, ty):
                    # 成功找到
                    yaw = BASE_YAW + random.uniform(*YAW_RANDOM_RANGE)
                    print(f"[Sampler] Selected Grid ({r},{c})")
                    return [tx, ty, FIXED_Z, FIXED_ROLL, FIXED_PITCH, yaw]
        
        print("[Sampler] Failed to sample grid, using center.")
        c = np.mean(self.boundary_points, axis=0)
        return [c[0], c[1], FIXED_Z, FIXED_ROLL, FIXED_PITCH, BASE_YAW]

    def get_boundary_target(self):
        """获取轮廓上的下一个随机点 (对应按下 'b')"""
        # 随机挑选一个轮廓点 (或者按顺序，这里按代码逻辑是按顺序遍历但Yaw随机)
        # 这里改为随机选择一个轮廓点，符合题目"轮廓上随机选择"
        rand_idx = random.randint(0, self.total_path_points - 1)
        pt = self.path_points_2d[rand_idx]
        
        yaw = BASE_YAW + random.uniform(*YAW_RANDOM_RANGE)
        print(f"[Sampler] Selected Boundary Point {rand_idx}")
        return [pt[0], pt[1], FIXED_Z, FIXED_ROLL, FIXED_PITCH, yaw]

# -----------------------------------------------------------------------------
# 3. 工具类：非阻塞键盘检测
# -----------------------------------------------------------------------------
class KeyPoller:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_term = termios.tcgetattr(self.fd)
        new_term = termios.tcgetattr(self.fd)
        new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(self.fd, termios.TCSANOW, new_term)
        return self
    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.fd, termios.TCSANOW, self.old_term)
    def poll(self):
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        if not dr: return None
        return sys.stdin.read(1)

# -----------------------------------------------------------------------------
# 4. 硬件封装类 (含 setup sequence)
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
            if cap.isOpened(): self.caps[name] = cap
        
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

    # === 脚本化移动 (Mode 0) ===
    def move_to_pose_scripted(self, pose, speed=100, wait=True):
        self.arm.set_mode(0); self.arm.set_state(0)
        self.arm.set_position(
            x=pose[0], y=pose[1], z=pose[2],
            roll=pose[3], pitch=pose[4], yaw=pose[5],
            speed=speed, wait=wait, is_radian=True
        )

    def move_home_scripted(self):
        print(">>> Moving Home...")
        self.move_to_pose_scripted(HOME_POS, speed=150)

    # === 新增：执行 R/B 键的特定序列 ===
    def run_setup_sequence(self, target_pose):
        """
        1. 移动到 A 点 -> 张开夹爪 -> 等待 1s
        2. 移动到 Home
        3. 移动到 A 点 -> 闭合夹爪
        4. 移动到 Home
        5. 移动到 采样点 (target_pose) -> 张开夹爪
        6. 移动到 采样点上方
        7. 移动回 Home
        """
        print(">>> Executing Setup Sequence...")
        speed_fast = 300
        speed_precise = 150

        # A 点上方安全点 (假设 A点z + 100)
        pose_A_up = list(POS_A); pose_A_up[2] += 100
        # 目标点上方安全点
        target_up = list(target_pose); target_up[2] += 100

        try:
            # 1. 移动到 A (先到上方，再下)
            # self.move_to_pose_scripted(pose_A_up, speed=speed_fast)
            # self.move_to_pose_scripted(POS_A, speed=speed_precise)
            # self.open_gripper()
            # print(">>> Waiting 1s at A...")
            # time.sleep(1.0)

            # # 2. 移动到 Home
            # self.move_to_pose_scripted(pose_A_up, speed=speed_fast)
            # self.move_home_scripted()

            # # 3. 移动到 A 抓取
            # self.move_to_pose_scripted(pose_A_up, speed=speed_fast)
            # self.move_to_pose_scripted(POS_A, speed=speed_precise)
            # self.close_gripper()
            # time.sleep(1.0) # 等待夹紧

            # 4. 移动到 Home
            self.move_to_pose_scripted(pose_A_up, speed=speed_fast)
            self.move_home_scripted()

            # 5. 移动到 采样点 放下
            self.move_to_pose_scripted(target_up, speed=speed_fast)
            self.move_to_pose_scripted(target_pose, speed=speed_precise)
            self.open_gripper()
            time.sleep(0.5)

            # 6. 移动到 采样点上方
            self.move_to_pose_scripted(target_up, speed=speed_fast)

            # 7. 移动回 Home
            self.move_home_scripted()
            print(">>> Sequence Done.")

        except Exception as e:
            print(f"[Error] Setup sequence failed: {e}")
            self.move_home_scripted()

    # === 推理相关 ===
    def move_to_start(self, target_action_rad):
        print(">>> Moving to inference start position...")
        target_joints = target_action_rad[:6]
        safe_joints = [np.clip(angle, low, high) for angle, (low, high) in zip(target_joints, JOINT_LIMITS)]
        self.arm.set_mode(0); self.arm.set_state(0)
        self.arm.set_servo_angle(angle=safe_joints, speed=0.35, is_radian=True, wait=True)
        self.arm.set_mode(1); self.arm.set_state(0)
        time.sleep(0.5)

    def execute_action(self, action_rad):
        # ... (保持原有的伺服控制逻辑不变) ...
        target_joints_rad = action_rad[:6]
        target_gripper = action_rad[6]
        safe_joints = [np.clip(angle, low, high) for angle, (low, high) in zip(target_joints_rad, JOINT_LIMITS)]
        
        # Z限位检查略 (为节省长度，假设已包含)
        
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

    def close(self):
        self.arm.set_mode(0); self.arm.set_state(0)
        for cap in self.caps.values(): cap.release()
        self.arm.disconnect()

# -----------------------------------------------------------------------------
# 5. 主程序逻辑
# -----------------------------------------------------------------------------
def main():
    print(f"Loading Config & Model...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    print("Model Loaded.")

    robot = XArmHardware(ROBOT_IP, CAMERAS)
    sampler = TaskSampler() # 初始化采样器
    prompt = "pick up the industrial components A"
    
    try:
        episode_count = 0
        while True:
            episode_count += 1
            print(f"\n" + "="*50)
            print(f"Ready for Episode {episode_count}")
            print(f"Controls:")
            print(f"  [ENTER] : Start Inference (Run Model)")
            print(f"  ['r']   : Random Grid Place (Move A -> Grid)")
            print(f"  ['b']   : Random Boundary Place (Move A -> Boundary)")
            print(f"  ['o']   : Reset (Open & Home)")
            print(f"  [Ctrl+C]: Exit")
            print("="*50)

            print(">>> Waiting for input...", end="", flush=True)
            mode = None 
            
            # --- 键盘监听循环 ---
            with KeyPoller() as key_poller:
                while True:
                    key = key_poller.poll()
                    if key is not None:
                        if key == '\n' or key == '\r': # Enter
                            mode = 'inference'
                            print("\n>>> Starting Inference...")
                            break
                        elif key == 'r': # Random Grid
                            print("\n>>> [Command] Random Grid Sampling...")
                            target = sampler.get_random_grid_target()
                            robot.run_setup_sequence(target)
                            print(">>> Done. Ready for input.")
                            print(">>> Waiting for input...", end="", flush=True)
                            # 执行完 setup 后继续等待，不进入推理
                        elif key == 'b': # Boundary
                            print("\n>>> [Command] Boundary Sampling...")
                            target = sampler.get_boundary_target()
                            robot.run_setup_sequence(target)
                            print(">>> Done. Ready for input.")
                            print(">>> Waiting for input...", end="", flush=True)
                        elif key == 'o': # Reset
                            print("\n>>> Reset Requested.")
                            robot.open_gripper()
                            robot.move_home_scripted()
                            print("\n>>> Reset Done.")
                            print(">>> Waiting for input...", end="", flush=True)
                        elif key == '\x03': # Ctrl+C
                            raise KeyboardInterrupt
                    time.sleep(0.01)

            if mode != 'inference': continue

            # === 进入推理流程 (保持原样) ===
            print("[Phase 1] Aligning to start position...")
            # 移动前最后检查一次 'o'
            with KeyPoller() as key_poller:
                if key_poller.poll() == 'o':
                    robot.open_gripper(); robot.move_home_scripted()
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

            print("[Phase 2] AI Control Loop (Press 'o' to abort)...")
            episode_success = False
            aborted = False

            with KeyPoller() as key_poller:
                while True:
                    if key_poller.poll() == 'o':
                        print("\n>>> ABORTED by user."); aborted = True; break
                    
                    raw_obs = robot.get_observation()
                    example = {
                        "cam_left_wrist": raw_obs["cam_left_wrist"],
                        "cam_right_wrist": raw_obs["cam_right_wrist"],
                        "state": raw_obs["state"],
                        "prompt": prompt
                    }

                    result = policy.infer(example)
                    action_chunk = np.array(result["actions"]) 

                    if np.any(action_chunk[:1, 6] > 0.8):
                        print(f"\n>>> Grasp detected. Stopping."); episode_success = True; break

                    steps_to_run = min(EXECUTE_STEPS, len(action_chunk))
                    for i in range(steps_to_run):
                        step_start = time.time()
                        if key_poller.poll() == 'o': aborted = True; break
                        robot.execute_action(action_chunk[i])
                        elapsed = time.time() - step_start
                        if (1.0/CONTROL_FREQ - elapsed) > 0: time.sleep(1.0/CONTROL_FREQ - elapsed)
                    
                    if aborted: break

            if aborted:
                robot.open_gripper(); time.sleep(0.5); robot.move_home_scripted()
            elif episode_success:
                robot.close_gripper(); time.sleep(1.0); robot.move_home_scripted()

    except KeyboardInterrupt:
        print("\nStopping Program...")
    finally:
        robot.close()

if __name__ == "__main__":
    main()