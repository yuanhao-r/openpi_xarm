import time
import cv2
import numpy as np
import sys
import os
import termios
import select
import random
import threading
from pathlib import Path
from scipy.spatial import ConvexHull
from xarm.wrapper import XArmAPI

# OpenPI 依赖
current_dir = os.path.dirname(os.path.abspath(__file__))
openpi_client_path = os.path.join(current_dir, "../../../packages/openpi-client/src")
sys.path.append("/home/openpi/src")
sys.path.append(os.path.abspath(openpi_client_path))

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

# -----------------------------------------------------------------------------
# 1. 配置区域 (保持 Code A 原样)
# -----------------------------------------------------------------------------
ROBOT_IP = "192.168.1.232"
CONFIG_NAME = "pi05_xarm_1212_night"
CHECKPOINT_DIR = "/home/openpi/checkpoints/exp10/58000"
VIS_SAVE_DIR = "/home/openpi/examples/xarm_real/images"

CAMERAS = {
    "cam_left_wrist": 0,
    "cam_right_wrist": 2
}
CROP_CONFIGS = {
    "cam_left_wrist": (118, 60, 357, 420),
    "cam_right_wrist": (136, 57, 349, 412)
}

CONTROL_FREQ = 10 
EXECUTE_STEPS = 4 
JOINT_LIMITS = [
    (-6.2, 6.2), (-2.0, 2.0), (-2.9, 2.9), 
    (-3.1, 3.1), (-1.6, 1.8), (-6.2, 6.2)
]

HOME_POS = [539.120605, 17.047951, 100-59.568863, 3.12897, 0.012689, -1.01436]
POS_A = [539.120605, 17.047951, -69.568863, 3.12897, 0.012689, -1.01436]
MIN_SAFE_Z = -69
SLOW_DOWN_FACTOR = 3.0  
INTERPOLATION_FREQ = 100.0 

# exp9 boundary
BOUNDARY_POINTS_2D = np.array([
   [505.982422, -150.631149],
   [712.302856, -66.848724],
   [697.232117, 163.981003],
   [466.805481, 144.618057],
])

FIXED_Z = POS_A[2]
FIXED_ROLL = POS_A[3]
FIXED_PITCH = POS_A[4]
BASE_YAW = POS_A[5]
YAW_RANDOM_RANGE = (-np.pi/6, np.pi/6)

# -----------------------------------------------------------------------------
# 键盘监听线程 (轻量级)
# -----------------------------------------------------------------------------
class KeyboardThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.last_key = None
        self.running = True

    def run(self):
        fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(fd)
        new_term = termios.tcgetattr(fd)
        new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(fd, termios.TCSANOW, new_term)
        try:
            while self.running:
                dr, _, _ = select.select([sys.stdin], [], [], 0.1)
                if dr:
                    self.last_key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old_term)

    def get_and_clear_key(self):
        k = self.last_key
        self.last_key = None
        return k

    def stop(self):
        self.running = False

# -----------------------------------------------------------------------------
# 工具类 (Sampler, Visualizer) - 保持不变
# -----------------------------------------------------------------------------
class TaskVisualizer:
    def __init__(self, save_dir, boundary_points, home_pos):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / "performance_map_exp10_58000_0105rainingDay_test2.png"
        self.boundary = boundary_points
        self.home_pos = home_pos
        self.scale = 1.5 
        self.offset_x = -np.min(boundary_points[:, 0]) * self.scale + 50
        self.offset_y = -np.min(boundary_points[:, 1]) * self.scale + 50
        self.canvas = self._load_or_create()

    def _to_pixel(self, x, y):
        return int(x * self.scale + self.offset_x), int(y * self.scale + self.offset_y)

    def _load_or_create(self):
        if self.save_path.exists(): return cv2.imread(str(self.save_path))
        w = int((np.max(self.boundary[:, 0]) - np.min(self.boundary[:, 0])) * self.scale + 100)
        h = int((np.max(self.boundary[:, 1]) - np.min(self.boundary[:, 1])) * self.scale + 100)
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        pts = np.array([self._to_pixel(p[0], p[1]) for p in self.boundary], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 0, 0), 2)
        cv2.circle(img, self._to_pixel(*self.home_pos[:2]), 8, (255, 0, 0), -1)
        return img

    def update_result(self, pose, success):
        pt = self._to_pixel(pose[0], pose[1])
        color = (0, 255, 0) if success else (0, 0, 255)
        end = (int(pt[0] + 25 * np.cos(pose[5])), int(pt[1] + 25 * np.sin(pose[5])))
        cv2.circle(self.canvas, pt, 4, color, -1)
        cv2.arrowedLine(self.canvas, pt, end, color, 2, tipLength=0.3)
        cv2.imwrite(str(self.save_path), self.canvas)

class TaskSampler:
    def __init__(self):
        self.hull = ConvexHull(BOUNDARY_POINTS_2D)
        # Grid 初始化
        self.min_x = np.min(BOUNDARY_POINTS_2D[:, 0])
        self.max_x = np.max(BOUNDARY_POINTS_2D[:, 0])
        self.min_y = np.min(BOUNDARY_POINTS_2D[:, 1])
        self.max_y = np.max(BOUNDARY_POINTS_2D[:, 1])
        self.grid_rows, self.grid_cols = 4, 4
        self.grid_indices = []
        self._refill()
        ccw_indices = self.hull.vertices
        self.ccw_vertices = BOUNDARY_POINTS_2D[ccw_indices]
        # 这里的 20.0 是步长，表示每隔 20mm 生成一个采样点
        self.path_points_2d = self._generate_boundary_path(self.ccw_vertices, 20.0)
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
                path.append(p_curr + unit_vec * (s * step_size))
        return np.array(path)
    def _refill(self):
        self.grid_indices = [(r, c) for r in range(self.grid_rows) for c in range(self.grid_cols)]
        random.shuffle(self.grid_indices)
    def is_inside(self, x, y):
        return all(np.dot(eq, [x, y, 1]) <= 1e-6 for eq in self.hull.equations)
    def get_target(self, mode='grid'):
        rand_yaw = BASE_YAW + random.uniform(YAW_RANDOM_RANGE[0], YAW_RANDOM_RANGE[1])
        if mode == 'boundary': # Simplified for brevity
            #  return [np.mean(BOUNDARY_POINTS_2D[:,0]), np.mean(BOUNDARY_POINTS_2D[:,1]), FIXED_Z, FIXED_ROLL, FIXED_PITCH, BASE_YAW]
            if len(self.path_points_2d) > 0:
                idx = random.randint(0, len(self.path_points_2d) - 1)
                pt = self.path_points_2d[idx]
                return [pt[0], pt[1], FIXED_Z, FIXED_ROLL, FIXED_PITCH, rand_yaw]
            else:
                # 保底逻辑
                c = np.mean(BOUNDARY_POINTS_2D, axis=0)
                return [c[0], c[1], FIXED_Z, FIXED_ROLL, FIXED_PITCH, rand_yaw]
        else: # Grid mode
            # 尝试多次采样直到在凸包内
            for _ in range(32):
                if not self.grid_indices: self._refill()
                r, c = self.grid_indices.pop()
                
                step_x = (self.max_x - self.min_x) / self.grid_cols
                step_y = (self.max_y - self.min_y) / self.grid_rows
                
                cell_min_x = self.min_x + c * step_x
                cell_max_x = self.min_x + (c + 1) * step_x
                cell_min_y = self.min_y + r * step_y
                cell_max_y = self.min_y + (r + 1) * step_y

                for _ in range(10): # 在格子内尝试几次
                    tx = random.uniform(cell_min_x, cell_max_x)
                    ty = random.uniform(cell_min_y, cell_max_y)
                    if self.is_inside(tx, ty):
                        # 注意这里使用的是 rand_yaw
                        return [tx, ty, FIXED_Z, FIXED_ROLL, FIXED_PITCH, rand_yaw]
            
            # 如果实在找不到，返回中心点 (保底)
            c = np.mean(BOUNDARY_POINTS_2D, axis=0)
            return [c[0], c[1], FIXED_Z, FIXED_ROLL, FIXED_PITCH, rand_yaw]
# -----------------------------------------------------------------------------
# 硬件封装 (核心修正：Flush Camera + Restore execute_action)
# -----------------------------------------------------------------------------
class XArmHardware:
    def __init__(self, ip, camera_indices):
        print(f"Connecting to xArm at {ip}...")
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0); self.arm.set_state(0)
        self.arm.set_tgpio_modbus_baudrate(baud=115200)
        
        self.caps = {}
        for name, idx in camera_indices.items():
            cap = cv2.VideoCapture(idx)
            cap.set(3, 640); cap.set(4, 480); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.caps[name] = cap
        
        self.current_gripper_state = 0.0
        self.open_gripper()
        time.sleep(1.5)

    def close_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 1.0

    def open_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])
        self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
        self.current_gripper_state = 0.0

    # 【新增】清空相机缓冲区，解决延迟问题的核心
    def flush_cameras(self):
        for cap in self.caps.values():
            for _ in range(4): # 连续读取几次，丢弃旧帧
                cap.grab()

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
        obs["state"] = np.append(joints_rad[:6], self.current_gripper_state)
        return obs

    # 【还原】完全恢复 Code A 的执行逻辑，去掉所有额外检测
    def execute_action(self, action_rad):
        # 1. 安全限位
        target_joints = [np.clip(a, l, h) for a, (l, h) in zip(action_rad[:6], JOINT_LIMITS)]
        
        # 2. Z轴检查 (Code A 有，所以这里加上，保持物理一致)
        ret, pose = self.arm.get_forward_kinematics(angles=target_joints, input_is_radian=True, return_is_radian=True)
        if ret == 0:
            if pose[2] < MIN_SAFE_Z:
                safe_pose = list(pose); safe_pose[2] = MIN_SAFE_Z
                ret_ik, ik_joints = self.arm.get_inverse_kinematics(safe_pose, input_is_radian=True, return_is_radian=True)
                if ret_ik == 0: target_joints = list(ik_joints)
                else: return # IK 失败跳过

        # 3. 插值运动 (Time Dilation)
        code, current_joints = self.arm.get_servo_angle(is_radian=True)
        if code != 0: return

        curr_j = np.array(current_joints[:6])
        targ_j = np.array(target_joints[:6])
        
        duration = (1.0 / CONTROL_FREQ) * SLOW_DOWN_FACTOR
        steps = int(duration * INTERPOLATION_FREQ)
        if steps < 1: steps = 1
        
        # === 核心循环：这里绝对不能有任何 IO 阻塞 ===
        for i in range(1, steps + 1):
            alpha = i / steps
            interp = curr_j + (targ_j - curr_j) * alpha
            self.arm.set_servo_angle_j(angles=np.append(interp, 0.0), is_radian=True)
            time.sleep(1.0 / INTERPOLATION_FREQ)
        # ==========================================

        # 4. 夹爪
        g = action_rad[6]
        if g > 0.8: self.close_gripper()
        elif g < 0.2: self.open_gripper()

    def move_home_scripted(self):
        self.arm.set_mode(0); self.arm.set_state(0)
        self.arm.set_position(*HOME_POS, speed=100, wait=True, is_radian=True)

    def move_to_start(self, target_action_rad):
        joints = [np.clip(a, l, h) for a, (l, h) in zip(target_action_rad[:6], JOINT_LIMITS)]
        self.arm.set_mode(0); self.arm.set_state(0)
        self.arm.set_servo_angle(angle=joints, speed=0.35, is_radian=True, wait=True)
        self.arm.set_mode(1); self.arm.set_state(0)
        time.sleep(0.5)

    def run_setup(self, target_pose):
        pose_A_up = list(POS_A); pose_A_up[2] += 100
        target_up = list(target_pose); target_up[2] += 100
        self.arm.set_mode(0); self.arm.set_state(0)
        try:
            self.arm.set_position(*pose_A_up, speed=300, wait=True, is_radian=True)
            self.arm.set_position(*POS_A, speed=300, wait=True, is_radian=True)
            
            self.close_gripper(); time.sleep(1.5)
            self.arm.set_position(*pose_A_up, speed=300, wait=True, is_radian=True)
            self.move_home_scripted()
            self.arm.set_position(*target_up, speed=300, wait=True, is_radian=True)
            self.arm.set_position(*target_pose, speed=300, wait=True, is_radian=True)
            self.open_gripper(); time.sleep(1.5)
            self.arm.set_position(*target_up, speed=300, wait=True, is_radian=True)
            self.move_home_scripted()
        except Exception: self.move_home_scripted()

    def close(self):
        self.arm.disconnect()
        for cap in self.caps.values(): cap.release()

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
def main():
    print(f"Loading Model: {CONFIG_NAME}...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    robot = XArmHardware(ROBOT_IP, CAMERAS)
    sampler = TaskSampler()
    viz = TaskVisualizer(VIS_SAVE_DIR, BOUNDARY_POINTS_2D, HOME_POS)
    
    # 启动后台键盘监听
    kb = KeyboardThread()
    kb.start()
    
    prompt = "pick up the industrial components"
    current_target = None
    
    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n=== Episode {episode} ===")
            print("[r]andom / [b]oundary / [o] reset / [ENTER] start")
            
            # --- 阶段 1: 设置与等待 ---
            # 这个阶段不需要实时性，可以随便阻塞
            mode = None
            while mode is None:
                key = kb.get_and_clear_key()
                if key == '\n' or key == '\r': mode = 'run'
                elif key == 'r': 
                    current_target = sampler.get_target('grid')
                    robot.run_setup(current_target)
                    print("Ready.")
                elif key == 'b':
                    current_target = sampler.get_target('boundary')
                    robot.run_setup(current_target)
                    print("Ready.")
                elif key == 'o':
                    robot.open_gripper(); robot.move_home_scripted()
                time.sleep(0.05)
            
            # --- 阶段 2: 准备推理 ---
            print(">>> Aligning...")
            
            # 【核心修复 1】: 在开始推理前，强制清空等待期间积压的相机帧
            robot.flush_cameras() 
            
            raw_obs = robot.get_observation()
            result = policy.infer({
                "cam_left_wrist": raw_obs["cam_left_wrist"],
                "cam_right_wrist": raw_obs["cam_right_wrist"],
                "state": raw_obs["state"], "prompt": prompt
            })
            robot.move_to_start(np.array(result["actions"])[0])
            
            # --- 阶段 3: AI 控制循环 (完全复刻 Code A) ---
            print(">>> AI Loop (Non-blocking). Press 'o' to STOP.")
            aborted = False
            
            while True:
                # 【核心修复 2】: 极速检查退出，不要用 select 阻塞
                if kb.get_and_clear_key() == 'o':
                    aborted = True; break
                
                # 1. 观测 (Code A: get_observation)
                raw_obs = robot.get_observation()
                
                # 2. 推理
                result = policy.infer({
                    "cam_left_wrist": raw_obs["cam_left_wrist"],
                    "cam_right_wrist": raw_obs["cam_right_wrist"],
                    "state": raw_obs["state"], "prompt": prompt
                })
                action_chunk = np.array(result["actions"])
                
                # 3. 抓取检测
                if np.any(action_chunk[:1, 6] > 0.8):
                    print(">>> Auto Grasp Detected.")
                    break
                
                # 4. 执行 (完全一致的循环结构)
                steps_to_run = min(EXECUTE_STEPS, len(action_chunk))
                for i in range(steps_to_run):
                    step_start = time.time()
                    
                    # 再次极速检查停止
                    if kb.get_and_clear_key() == 'o': 
                        aborted = True; break
                    
                    # 执行动作 (内部含插值，耗时约 0.3s)
                    robot.execute_action(action_chunk[i])
                    
                    # 频率控制 (Code A 逻辑)
                    # 由于 execute_action 耗时 > 1/CONTROL_FREQ，这里 sleep_time 通常为负，不会 sleep
                    elapsed = time.time() - step_start
                    sleep_time = (1.0 / CONTROL_FREQ) - elapsed
                    if sleep_time > 0: time.sleep(sleep_time)
                
                if aborted: break
            
            # --- 阶段 4: 结算 ---
            if current_target:
                print(">>> Success? [y/n]")
                while True:
                    k = kb.get_and_clear_key()
                    if k == 'y': viz.update_result(current_target, True); break
                    elif k == 'n': viz.update_result(current_target, False); break
                    time.sleep(0.05)
                current_target = None
            
            if aborted: robot.open_gripper()
            else: robot.close_gripper()
            time.sleep(1.5)
            robot.move_home_scripted()

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        kb.stop()
        robot.close()

if __name__ == "__main__":
    main()