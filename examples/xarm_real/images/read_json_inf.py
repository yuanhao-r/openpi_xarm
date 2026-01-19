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
import json

# OpenPI 依赖
current_dir = os.path.dirname(os.path.abspath(__file__))
openpi_client_path = os.path.join(current_dir, "../../../packages/openpi-client/src")
sys.path.append("/home/openpi/src")
sys.path.append(os.path.abspath(openpi_client_path))

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

import matplotlib
matplotlib.use('Agg') # 设置后端为非交互式，Docker 专用
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. 配置区域 (保持 Code A 原样)
# -----------------------------------------------------------------------------
ROBOT_IP = "192.168.1.232"
CONFIG_NAME = "pi05_xarm_1212_night"
CHECKPOINT_DIR = "/home/openpi/checkpoints/exp17/24000"
VIS_SAVE_DIR = "/home/openpi/examples/xarm_real/images"
RESULT_IMG_NAME = "0113morning_sunshineDay_exp17sub_30000_test1(components C).png"
TASK_PROMOT = "pick up the industrial components"
#指定要读取的点位文件
POINTS_FILE = os.path.join(VIS_SAVE_DIR, "test_points.json")

CAMERAS = {
    "cam_left_wrist": "/dev/cam_left_wrist",
    "cam_right_wrist": "/dev/cam_right_wrist"
}
CROP_CONFIGS = {
    "cam_left_wrist": (118, 60, 357, 420),
    "cam_right_wrist": (136, 57, 349, 412)
}

CONTROL_FREQ = 10 
EXECUTE_STEPS = 2
JOINT_LIMITS = [
    (-6.2, 6.2), (-2.0, 2.0), (-2.9, 2.9), 
    (-3.1, 3.1), (-1.6, 1.8), (-6.2, 6.2)
]

HOME_POS = [486.626923, 158.343277, 30.431152, 3.12897, 0.012689, -1.01436]
POS_A = [486.626923, 158.343277, -79, 3.12897, 0.012689, -1.01436]
MIN_SAFE_Z = -99
# HOME_POS = [539.120605, 17.047951, 100-59.568863, 3.12897, 0.012689, -1.01436]
# POS_A = [539.120605, 17.047951, -79.568863, 3.12897, 0.012689, -1.01436]
# MIN_SAFE_Z = -99
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
# 【修复版】Docker 专用无头可视化器
# -----------------------------------------------------------------------------
class DebugVisualizer:
    def __init__(self, safe_z_limit, save_dir):
        # 增加画布高度，改为 3行 2列
        self.fig, self.axs = plt.subplots(3, 2, figsize=(10, 12))
        self.safe_z_limit = safe_z_limit
        self.save_path = os.path.join(save_dir, "live_debug_status.png")
        
        # --- 布局定义 ---
        # Row 1: 相机
        self.ax_cam1 = self.axs[0, 0]
        self.ax_cam2 = self.axs[0, 1]
        
        # Row 2: 空间轨迹 (左: XY平面, 右: Z高度)
        self.ax_xy_plane = self.axs[1, 0]
        self.ax_z = self.axs[1, 1]
        
        # Row 3: 数值曲线 (左: XY随时间变化, 右: 夹爪)
        self.ax_xy_time = self.axs[2, 0]
        self.ax_grip = self.axs[2, 1]
        
        # --- 初始化样式 ---
        # 1. XY 平面 (俯视图)
        self.ax_xy_plane.set_title("XY Trajectory (Top-Down View)")
        self.ax_xy_plane.set_xlabel("X (mm)")
        self.ax_xy_plane.set_ylabel("Y (mm)")
        self.ax_xy_plane.grid(True)
        self.ax_xy_plane.set_aspect('equal', 'datalim') # 保持比例，防止圆形变椭圆
        
        # 2. Z 轴
        self.ax_z.set_title("Z Trajectory (Height)")
        self.ax_z.set_ylabel("Z (mm)")
        self.ax_z.axhline(y=safe_z_limit, color='r', linestyle='--', label='Limit')
        self.ax_z.grid(True)
        
        # 3. XY 时间序列
        self.ax_xy_time.set_title("X & Y over Time (Steps)")
        self.ax_xy_time.set_ylabel("Position (mm)")
        self.ax_xy_time.grid(True)
        
        # 4. 夹爪
        self.ax_grip.set_title("Gripper Intent")
        self.ax_grip.set_ylim(-0.1, 1.1)
        self.ax_grip.axhline(y=0.8, color='g', linestyle='--', label='Trigger')
        self.ax_grip.grid(True)

        print(f"[Vis] Debug visualization will be saved to: {self.save_path}")

    def _clear_lines(self, ax):
        """辅助函数：安全清除图表中的线条"""
        for line in list(ax.lines):
            line.remove()
        # 清除图例
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    def update(self, obs, action_chunk, robot_arm):
        # --- 1. 绘制图像 ---
        self.ax_cam1.clear(); self.ax_cam1.set_title("Left Wrist")
        self.ax_cam1.imshow(obs['cam_left_wrist'])
        self.ax_cam1.axis('off')
        
        self.ax_cam2.clear(); self.ax_cam2.set_title("Right Wrist")
        self.ax_cam2.imshow(obs['cam_right_wrist'])
        self.ax_cam2.axis('off')
        
        # --- 2. 计算轨迹数据 (FK) ---
        pred_x, pred_y, pred_z = [], [], []
        
        for i in range(len(action_chunk)):
            joints = action_chunk[i][:6]
            # 计算正运动学
            ret, pose = robot_arm.get_forward_kinematics(angles=joints, input_is_radian=True, return_is_radian=True)
            if ret == 0: 
                pred_x.append(pose[0])
                pred_y.append(pose[1])
                pred_z.append(pose[2])
            else: 
                # 如果解算失败，沿用上一个点或补0
                last_x = pred_x[-1] if pred_x else 0
                last_y = pred_y[-1] if pred_y else 0
                last_z = pred_z[-1] if pred_z else 0
                pred_x.append(last_x); pred_y.append(last_y); pred_z.append(last_z)
        
        steps = np.arange(len(pred_x)) # 时间步

        # --- 3. 绘制 XY 平面轨迹 (俯视图) ---
        self._clear_lines(self.ax_xy_plane)
        # 画轨迹线
        self.ax_xy_plane.plot(pred_x, pred_y, 'b-o', alpha=0.6, markersize=4, label='Path')
        # 标记起点 (绿色) 和 终点 (红色)
        if pred_x:
            self.ax_xy_plane.plot(pred_x[0], pred_y[0], 'go', markersize=8, label='Start') # 起点
            self.ax_xy_plane.plot(pred_x[-1], pred_y[-1], 'rx', markersize=8, label='End') # 终点
            
            # 动态调整视野，保证能看清轨迹趋势 (加一点padding)
            mid_x, span_x = (np.min(pred_x) + np.max(pred_x))/2, (np.max(pred_x) - np.min(pred_x))
            mid_y, span_y = (np.min(pred_y) + np.max(pred_y))/2, (np.max(pred_y) - np.min(pred_y))
            max_span = max(span_x, span_y, 20) # 最小视野20mm
            self.ax_xy_plane.set_xlim(mid_x - max_span, mid_x + max_span)
            self.ax_xy_plane.set_ylim(mid_y - max_span, mid_y + max_span)
            
        self.ax_xy_plane.legend(loc='upper right', fontsize='small')

        # --- 4. 绘制 Z 轴高度 ---
        self._clear_lines(self.ax_z)
        self.ax_z.axhline(y=self.safe_z_limit, color='r', linestyle='--')
        self.ax_z.plot(steps, pred_z, 'b-o', markersize=4)
        # 动态调整 Z 轴范围，方便看清是否贴地
        if pred_z:
            min_z = min(min(pred_z), self.safe_z_limit)
            self.ax_z.set_ylim(min_z - 20, max(pred_z) + 20)

        # --- 5. 绘制 XY 时间序列 (看震荡) ---
        self._clear_lines(self.ax_xy_time)
        self.ax_xy_time.plot(steps, pred_x, 'c--', label='X')
        self.ax_xy_time.plot(steps, pred_y, 'm--', label='Y')
        self.ax_xy_time.legend(loc='best', fontsize='small')

        # --- 6. 绘制夹爪 ---
        grip_vals = action_chunk[:, 6]
        self._clear_lines(self.ax_grip)
        self.ax_grip.axhline(y=0.8, color='g', linestyle='--')
        self.ax_grip.plot(steps, grip_vals, 'k-x')

        # --- 7. 保存图片 (Docker 兼容) ---
        try:
            self.fig.canvas.draw()
            img_rgba = np.asarray(self.fig.canvas.buffer_rgba())
            image = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(self.save_path, image)
        except Exception as e:
            print(f"[Vis Error] {e}")
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
    def __init__(self, save_dir, result_name, boundary_points, home_pos):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / result_name
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
        save_path_str = str(self.save_path)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if not (save_path_str.endswith(".png") or save_path_str.endswith(".jpg")):
            print(f"[Warn] Filename '{save_path_str}' has no valid extension. Appending .png")
            self.save_path = self.save_path.with_suffix(".png")
            save_path_str = str(self.save_path)
        # 打印调试信息 (看看究竟存到哪去了)
        print(f"[Debug] Saving image to: {save_path_str}")
        
        try:
            cv2.imwrite(save_path_str, self.canvas)
            print(f"[Vis] Result saved successfully.")
        except Exception as e:
            print(f"[Error] Failed to save image: {e}")

class TaskSampler:
    def __init__(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到点位文件: {json_path}。请先运行生成脚本。")
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 将 Grid 和 Boundary 的点合并成一个列表，按顺序执行
        # 如果你想先测 Boundary，可以把顺序反过来
        self.all_points = data["grid"] + data["boundary"]
        self.total_count = len(self.all_points)
        self.current_idx = 0
        
        print(f"[Sampler] Loaded {self.total_count} points (Grid + Boundary).")
    
    def get_next_target(self):
        """获取下一个点，如果测完了返回 None"""
        if self.current_idx < self.total_count:
            pose = self.all_points[self.current_idx]
            self.current_idx += 1
            return pose, self.current_idx, self.total_count
        return None, -1, self.total_count
    
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
        # self.open_gripper()
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
        
        # 2. Z轴检查与修正
        ret, pose = self.arm.get_forward_kinematics(angles=target_joints, input_is_radian=True, return_is_radian=True)
        if ret == 0:
            model_z = pose[2]
            
            # 只有当模型想去的高度 低于 安全限制时，才触发修正
            if model_z < MIN_SAFE_Z:
                print(f"[DEBUG] Z Limit Triggered! Model wants: {model_z:.2f}, Limit: {MIN_SAFE_Z}")
                
                # 构造修正后的位姿（保持XY和旋转不变，只把Z抬高到安全线）
                safe_pose = list(pose)
                safe_pose[2] = MIN_SAFE_Z
                
                # 重新解算关节角
                ret_ik, ik_joints = self.arm.get_inverse_kinematics(safe_pose, input_is_radian=True, return_is_radian=True)
                if ret_ik == 0: 
                    target_joints = list(ik_joints) # 使用修正后的关节角
                else: 
                    print("[Error] IK Failed during Z-safety adjustment")
                    return # IK 失败跳过该步骤

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
            
            self.close_gripper(); time.sleep(2.0)
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
    sampler = TaskSampler(POINTS_FILE)
    viz = TaskVisualizer(VIS_SAVE_DIR, RESULT_IMG_NAME, BOUNDARY_POINTS_2D, HOME_POS)
    debugger = DebugVisualizer(MIN_SAFE_Z, VIS_SAVE_DIR)

    
    # 启动后台键盘监听
    kb = KeyboardThread()
    kb.start()
    
    # prompt = "pick up the industrial components B"
    current_target = None
    
    try:
        episode = 0
        while True:
            episode += 1
            print(f"\n=== Episode {episode} ===")
            # 1. 获取下一个目标点
            target_pose, idx, total = sampler.get_next_target()
            # 如果没有点了，结束程序
            if target_pose is None:
                print("\n" + "="*50)
                print("ALL TEST POINTS COMPLETED!")
                print(f"Result image saved at: {viz.save_path}")
                print("="*50)
                break
            print(f"\n=== Test Point {idx}/{total} ===")
            print(f"Target: {target_pose}")
            
            # 2. 机器人去目标点放置物体 (Setup)
            robot.run_setup(target_pose)
            print(">>> Setup Done. Starting Inference...")            
            # 3. 推理准备
            robot.flush_cameras() 
            raw_obs = robot.get_observation()
            result = policy.infer({
                "cam_left_wrist": raw_obs["cam_left_wrist"],
                "cam_right_wrist": raw_obs["cam_right_wrist"],
                "state": raw_obs["state"], "prompt": TASK_PROMOT
            })
            robot.move_to_start(np.array(result["actions"])[0])
            
            
            
            
            
            # 4. AI 控制循环
            
            print(">>> AI Loop running... Press 'o' to ABORT (Mark as Fail).")
            aborted = False
            
            while True:
                #  极速检查退出，不要用 select 阻塞
                if kb.get_and_clear_key() == 'o':
                    aborted = True; break
                
                # 1. 观测 (Code A: get_observation)
                raw_obs = robot.get_observation()
                
                # 2. 推理
                result = policy.infer({
                    "cam_left_wrist": raw_obs["cam_left_wrist"],
                    "cam_right_wrist": raw_obs["cam_right_wrist"],
                    "state": raw_obs["state"], "prompt": TASK_PROMOT
                })
                action_chunk = np.array(result["actions"])
                
                debugger.update(raw_obs, action_chunk, robot.arm)
                
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
            
            # 5. 结算环节
            if aborted:
                print("\n>>> Inference Aborted by 'o'. Marking as FAILURE.")
                viz.update_result(target_pose, False) # 记为失败
                robot.open_gripper()
                time.sleep(1.5)
                robot.move_home_scripted()
            else:
                # 正常结束，等待人工判定
                robot.close_gripper()
                time.sleep(2.0)
                robot.move_home_scripted()
                print(">>> Marked as SUCCESS.")
                viz.update_result(target_pose, True)
                # print("\n>>> Evaluate Result: [y] Success / [n] Failure")
                # # 循环等待直到按下 y 或 n
                # while True:
                #     k = kb.get_and_clear_key()
                #     if k == 'y': 
                #         print(">>> Marked as SUCCESS.")
                #         viz.update_result(target_pose, True)
                #         break
                #     elif k == 'n': 
                #         print(">>> Marked as FAILURE.")
                #         viz.update_result(target_pose, False)
                #         break
                #     time.sleep(0.05)
                
    

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        kb.stop()
        robot.close()

if __name__ == "__main__":
    main()