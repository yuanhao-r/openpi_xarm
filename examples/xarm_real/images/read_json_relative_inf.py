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
CHECKPOINT_DIR = "/home/openpi/checkpoints/exp23/24000"
VIS_SAVE_DIR = "/home/openpi/examples/xarm_real/images"
# RESULT_IMG_NAME = "0121mafternoon_cloudyDay_exp22_36000_test20_components E23.png"
# TASK_PROMOT = "pick up the industrial components"
TASK_PROMOT = "pick up the small upright valve "
SELECTED_TASK = "D"  
# --- 任务配置字典 ---
TASK_CONFIGS = {
    "A": {
        "prompt": "pick up the hollow rectangular housing",
        "z": -65
    },
    "B": {
        "prompt": "pick up the silver metal cylinder",
        "z": -65
    },
    "C": {
        "prompt": "pick up the small upright valve",
        "z": -79
    },
    "D": {
        "prompt": "pick up the flat triangular plate",
        "z": -65
    },
    "E": {
        "prompt": "pick up the silver metal cylinder",
        "z": -65
    },
    "F": {
        "prompt": "pick up the flat circular plate",
        "z": -103
    },
    "G": {
        "prompt": "pick up the flat circular plate",
        "z": -109
    }
}
# 获取当前任务的配置
current_config = TASK_CONFIGS[SELECTED_TASK]
OBJECT_Z = current_config["z"]
# 设置提示词
TASK_PROMOT = current_config["prompt"]
# 基础坐标 (X, Y, Roll, Pitch, Yaw)
_base_x = 554.626923
_base_y = 361.343277
_base_r = 3.12897
_base_p = 0.012689
_base_yw = -1.01436
# 组装 POS_A (动态填入对应的 Z 值)
POS_A = [_base_x, _base_y, current_config["z"], _base_r, _base_p, _base_yw]
# 自动生成结果图片文件名 (避免手动改名)
RESULT_IMG_NAME = f"0127afternoon_cloudyDay_exp23_24000_test2_components_{SELECTED_TASK}.png"
print(f">>> 当前任务: [{SELECTED_TASK}]")
print(f">>> Prompt: {TASK_PROMOT}")
print(f">>> POS_A Z-height: {POS_A[2]}")
#指定要读取的点位文件
POINTS_FILE = os.path.join(VIS_SAVE_DIR, "test_points.json")
RESUME_TESTING = True # 开关：是否开启断点续测
PROGRESS_FILE = os.path.join(VIS_SAVE_DIR, "test_progress.json") # 进度文件路径
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

# HOME_POS = [486.626923, 297.343277, 30.431152, 3.12897, 0.012689, -1.01436]
# POS_A = [486.626923, 297.343277, -65, 3.12897, 0.012689, -1.01436]
HOME_POS = [554.626923, 361.343277, 30.431152, 3.12897, 0.012689, -1.01436]
# POS_A = [554.626923, 361.343277, -79, 3.12897, 0.012689, -1.01436]
MIN_SAFE_Z = -119
# HOME_POS = [539.120605, 17.047951, 100-59.568863, 3.12897, 0.012689, -1.01436]
# POS_A = [539.120605, 17.047951, -79.568863, 3.12897, 0.012689, -1.01436]
# MIN_SAFE_Z = -99
SLOW_DOWN_FACTOR = 2.0  
INTERPOLATION_FREQ = 100.0 

# exp9 boundary
BOUNDARY_POINTS_2D = np.array([
    [528.6, 126.5],
    [745.0, 250.2],
    [501.9, 539.4],
    [338.1, 425.0],
])
# 计算宽松边界 (Relaxed Boundary)
# 以中心点为基准，向外扩张 1.1 倍 (即允许超出 10%)
# 1.15 表示允许向外扩 15% 的范围，你可以根据实际桌子大小调整这个系数
_center_point = np.mean(BOUNDARY_POINTS_2D, axis=0)
BOUNDARY_EXPANDED = _center_point + (BOUNDARY_POINTS_2D - _center_point) * 1.15

FIXED_Z = POS_A[2]
FIXED_ROLL = POS_A[3]
FIXED_PITCH = POS_A[4]
BASE_YAW = POS_A[5]
YAW_RANDOM_RANGE = (-np.pi/6, np.pi/6)


class MetricsRecorder:
    def __init__(self):
        self.episode_metrics = []
        self.current_episode = {}
        
    def start_episode(self, start_pose_abs, ground_truth_pose):
        """
        开始记录一轮测试
        :param start_pose_abs: 机械臂起始的绝对坐标 [x, y, z, r, p, y] (单位: 米)
        :param ground_truth_pose: JSON文件里的目标坐标 [x, y, z, r, p, y] (单位: 毫米)
        """
        self.current_episode = {
            "start_time": time.time(),
            # 统一转换为毫米 (mm) 存储，方便计算
            "start_pose": np.array(start_pose_abs) * 1000.0, 
            "ground_truth_pose": np.array(ground_truth_pose), # 假设 JSON 里是 mm
            "trajectory": [], 
            "success": False,
            "steps": 0
        }
        # 记录起点
        self.current_episode["trajectory"].append(self.current_episode["start_pose"][:3])
        # =======================================================
        # 初始化 final_pos_mm 和 final_rpy_rad
        # 即使一步没走，当前的最终位置就是起始位置
        # =======================================================
        self.current_episode["final_pos_mm"] = self.current_episode["start_pose"][:3]
        self.current_episode["final_rpy_rad"] = start_pose_abs[3:]
    def step(self, current_pose_abs):
        """记录每一步的实际位置 (输入单位: 米)"""
        # 转为 mm 存储
        pos_mm = np.array(current_pose_abs[:3]) * 1000.0
        self.current_episode["trajectory"].append(pos_mm)
        self.current_episode["steps"] += 1
        
        # 实时更新最终位姿 (以最后一步为准)
        # 同时记录最后一步的旋转 (Roll, Pitch, Yaw) 用于算角度误差
        self.current_episode["final_full_pose"] = np.array(current_pose_abs) * 1000.0 # [x,y,z, r,p,y]
        # 注意：r,p,y 这里也被乘了1000，后面计算时要还原回去，或者分开处理
        # 修正：我们分开存
        self.current_episode["final_pos_mm"] = pos_mm
        self.current_episode["final_rpy_rad"] = current_pose_abs[3:]

    def end_episode(self, success, close_gripper_time):
        # self.current_episode["end_time"] = time.time()
        self.current_episode["end_time"] = close_gripper_time
        self.current_episode["success"] = success
        
        # 计算该轮指标
        metrics = self._calculate_single_metrics(self.current_episode)
        self.episode_metrics.append(metrics)
        return metrics

    def _calculate_single_metrics(self, data):
        # 1. 耗时
        duration = data["end_time"] - data["start_time"]
        
        # 2. 轨迹平滑度 (Jerk)
        traj = np.array(data["trajectory"]) # (T, 3) mm
        if len(traj) > 3:
            vel = np.diff(traj, axis=0)
            acc = np.diff(vel, axis=0)
            jerk = np.diff(acc, axis=0)
            avg_jerk = np.mean(np.linalg.norm(jerk, axis=1))
        else:
            avg_jerk = 0.0

        # 3. 步数效率 (基于起始点到理论目标点的距离)
        # Distance (Start -> Ground Truth)
        gt_pos = data["ground_truth_pose"][:3]
        start_pos = data["start_pose"][:3]
        dist_xy = np.linalg.norm(gt_pos[:2] - start_pos[:2])
        dist_z = abs(gt_pos[2] - start_pos[2])
        ideal_path_len = dist_xy + dist_z # mm
        
        # 定义“标准步长” 在训练数据集中，每一步(0.1s)平均移动多少毫米
        REF_STEP_LEN_MM = 5.0 
        
                
        # 假设理论速度 100mm/s, 10Hz -> 10mm/step
        # 避免除以0
        opt_steps = max(1, int(ideal_path_len / REF_STEP_LEN_MM))
        step_ratio = data["steps"] / opt_steps

        # 4. 最终误差 (核心指标)
        final_pos = data["final_pos_mm"]
        
        # A. 位置误差 (mm)
        pos_error = np.linalg.norm(final_pos - gt_pos)
        
        # B. 角度误差 (degree)
        # Ground Truth 的 RPY (假设 JSON 后三位是 rad)
        gt_rpy = data["ground_truth_pose"][3:] 
        final_rpy = data["final_rpy_rad"]
        
        # 简单计算 RPY 的欧氏距离作为误差参考 (更严谨可以用四元数)
        # 将弧度转为角度计算差值
        diff_rpy_deg = np.degrees(np.abs(final_rpy - gt_rpy))
        # 处理周期性 (例如 359度 和 1度 差2度) - 简单场景可忽略，这里做个简化求和
        rot_error = np.sum(diff_rpy_deg) # 累计角度误差

        return {
            "success": 1.0 if data["success"] else 0.0,
            "time": duration,
            "jerk": avg_jerk,
            "step_ratio": step_ratio,
            "pos_error": pos_error,
            "rot_error": rot_error,
            "opt_steps": opt_steps  
        }

    def print_summary(self):
        if not self.episode_metrics:
            print("No metrics data.")
            return

        N = len(self.episode_metrics)
        total_episodes = len(self.episode_metrics)
        success_list = [m for m in self.episode_metrics if m["success"] == 1.0]
        num_success = len(success_list)
        avg_success = (num_success / total_episodes) * 100.0
        # 3. 计算其他指标 (仅基于成功案例)
        if num_success > 0:
            avg_time = np.mean([m["time"] for m in success_list])
            avg_jerk = np.mean([m["jerk"] for m in success_list])
            avg_step_ratio = np.mean([m["step_ratio"] for m in success_list])
            avg_pos_error = np.mean([m["pos_error"] for m in success_list])
            avg_rot_error = np.mean([m["rot_error"] for m in success_list])
            
            # 也可以算一下标准差(std)看稳定性，这里先只展示均值
        else:
            # 如果一次都没成功，其他指标没有意义
            avg_time = 0.0
            avg_jerk = 0.0
            avg_step_ratio = 0.0
            avg_pos_error = 0.0
            avg_rot_error = 0.0
            
        # avg_success = np.mean([m["success"] for m in self.episode_metrics]) * 100.0
        # avg_time = np.mean([m["time"] for m in self.episode_metrics])
        # avg_jerk = np.mean([m["jerk"] for m in self.episode_metrics])
        # avg_step_ratio = np.mean([m["step_ratio"] for m in self.episode_metrics])
        # avg_pos_error = np.mean([m["pos_error"] for m in self.episode_metrics])
        # avg_rot_error = np.mean([m["rot_error"] for m in self.episode_metrics])

        if num_success > 0:
            print("\n" + "="*60)
            print(f"📊 量化测试报告 (已测: {N} 轮)")
            print("="*60)
            print(f"✅ 成功率 (Success Rate):     {avg_success:.1f}%")
            print(f"🎯 平均位置误差 (Pos Error): {avg_pos_error:.2f} mm")
            print(f"📐 平均角度误差 (Rot Error): {avg_rot_error:.2f} deg")
            print(f"⏱️ 平均耗时 (Time):          {avg_time:.2f} s")
            print(f"👣 步数比 (Actual/Optimal):   {avg_step_ratio:.2f}")
            print(f"📉 轨迹平滑度 (Jerk):        {avg_jerk:.4f}")
            print("="*60 + "\n")
        else:
            print("-" * 40)
            print("  [无成功案例，无法计算性能指标]")
            
    def print_current_metrics(self, metrics):
        """打印当前这一轮的详细指标"""
        status_str = "✅ Success" if metrics["success"] == 1.0 else "❌ Failure"
        
        print("-" * 40)
        print(f"📝 本轮详细数据 ({status_str})")
        print(f"   ⏱️ 耗时:       {metrics['time']:.2f} s")
        print(f"   🎯 位置误差:   {metrics['pos_error']:.2f} mm")
        print(f"   📐 角度误差:   {metrics['rot_error']:.2f} deg")
        print(f"   👣 步数比:     {metrics['step_ratio']:.2f} (Opt: {metrics['opt_steps']:.0f})")
        print(f"   📉 平滑度:     {metrics['jerk']:.4f}")
        print("-" * 40)
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
        # self.ax_xy_plane.set_aspect('equal', 'datalim') # 保持比例，防止圆形变椭圆
        self.ax_xy_plane.set_aspect('equal', adjustable='box')
        
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
        
        # 1. 获取当前机械臂的绝对笛卡尔坐标 (作为起点)
        # 注意：这里需要调用 robot_arm 的 get_position，注意单位转换
        code, curr_pose = robot_arm.get_position(is_radian=True)
        if code != 0: return # 读不到就算了
        curr_x, curr_y, curr_z = curr_pose[0], curr_pose[1], curr_pose[2]
        # --- 2. 计算轨迹数据 (FK) ---
        pred_x, pred_y, pred_z = [], [], []
        # 记录起始点的绝对坐标
        start_x, start_y, start_z = curr_x, curr_y, curr_z
        # 累计推演
        sim_x, sim_y, sim_z = curr_x, curr_y, curr_z
        for i in range(len(action_chunk)):
            # action_chunk[i] 是 [dx, dy, dz, ...] (单位: 米)
            dx = action_chunk[i][0] * 1000.0 # 转毫米
            dy = action_chunk[i][1] * 1000.0
            dz = action_chunk[i][2] * 1000.0
            
            cur_pred_x = start_x + dx
            cur_pred_y = start_y + dy
            cur_pred_z = start_z + dz
            
            pred_x.append(cur_pred_x)
            pred_y.append(cur_pred_y)
            pred_z.append(cur_pred_z)
        
       
        steps = np.arange(len(pred_x)) # 时间步

        # --- 3. 绘制 XY 平面轨迹 (俯视图) ---
        self._clear_lines(self.ax_xy_plane)
        self.ax_xy_plane.plot(pred_x, pred_y, 'b-o', alpha=0.6, markersize=4, label='Path')
        if pred_x:
            self.ax_xy_plane.plot(pred_x[0], pred_y[0], 'go', markersize=8, label='Start')
            self.ax_xy_plane.plot(pred_x[-1], pred_y[-1], 'rx', markersize=8, label='End')
            
            # 动态调整视野
            mid_x, span_x = (np.min(pred_x) + np.max(pred_x))/2, (np.max(pred_x) - np.min(pred_x))
            mid_y, span_y = (np.min(pred_y) + np.max(pred_y))/2, (np.max(pred_y) - np.min(pred_y))
            max_span = max(span_x, span_y, 20) 
            self.ax_xy_plane.set_xlim(mid_x - max_span, mid_x + max_span)
            self.ax_xy_plane.set_ylim(mid_y - max_span, mid_y + max_span)
            
        self.ax_xy_plane.legend(loc='upper right', fontsize='small')

        # --- 4. 绘制 Z 轴高度 ---
        self._clear_lines(self.ax_z)
        self.ax_z.axhline(y=self.safe_z_limit, color='r', linestyle='--') # 限位线
        self.ax_z.plot(steps, pred_z, 'b-o', markersize=4)
        # 动态调整 Z 轴范围，方便看清是否贴地
        if pred_z:
            min_z = min(min(pred_z), self.safe_z_limit)
            self.ax_z.set_ylim(min_z - 20, max(pred_z) + 20)

        # --- 5. 绘制 XY 时间序列 ---
        self._clear_lines(self.ax_xy_time)
        self.ax_xy_time.plot(steps, pred_x, 'c--', label='X')
        self.ax_xy_time.plot(steps, pred_y, 'm--', label='Y')
        self.ax_xy_time.legend(loc='best', fontsize='small')

        # --- 6. 绘制夹爪 (保持不变) ---
        grip_vals = action_chunk[:, 6]
        self._clear_lines(self.ax_grip)
        self.ax_grip.axhline(y=0.8, color='g', linestyle='--')
        self.ax_grip.plot(steps, grip_vals, 'k-x')

        # --- 7. 保存图片 ---
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
    def __init__(self, json_path, progress_file=None, resume=False):
        """
        :param json_path: 原始完整测试点文件 (test_points.json)
        :param progress_file: 进度记录文件路径 (test_progress.json)
        :param resume: 是否尝试从进度文件恢复
        """
        self.original_json_path = json_path
        self.progress_file = progress_file
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到点位文件: {json_path}。请先运行生成脚本。")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            # 原始全集（顺序必须固定）
            self.all_points_original = data["grid"] + data["boundary"]
        # 2. 处理断点恢复逻辑
        if resume and progress_file and os.path.exists(progress_file):
            print(f"[Sampler] 发现进度文件: {progress_file}，尝试恢复...")
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                # 读取剩余点列表
                self.remaining_points = progress_data.get("remaining_points", [])
                self.completed_count = progress_data.get("completed_count", 0)
                
                # 校验一下 (可选)
                if not self.remaining_points and self.completed_count > 0:
                    print("[Sampler] ⚠️ 进度文件显示所有点已测试完毕！")
                else:
                    print(f"[Sampler] ✅ 成功恢复进度。已测: {self.completed_count}, 剩余: {len(self.remaining_points)}")
            except Exception as e:
                print(f"[Sampler] ❌ 读取进度文件失败 ({e})，将重置为全部测试点。")
                self.remaining_points = list(self.all_points_original)
                self.completed_count = 0
        else:
            # 不续测，或者没有进度文件 -> 重置
            print("[Sampler] 初始化新测试序列...")
            self.remaining_points = list(self.all_points_original)
            self.completed_count = 0
        
            # 如果开启了续测模式但文件不存在，立刻创建一个初始状态
            if resume and progress_file:
                self.save_progress()
        self.total_original_count = len(self.all_points_original)
        self.current_target = None
                
        # 将 Grid 和 Boundary 的点合并成一个列表，按顺序执行
        # 如果你想先测 Boundary，可以把顺序反过来
        # self.all_points = data["grid"] + data["boundary"]
        # self.total_count = len(self.all_points)
        self.current_idx = 0
        
        print(f"[Sampler] Loaded {self.total_original_count} points (Grid + Boundary).")
    
    def get_next_target(self):
        """获取下一个点，并从剩余列表中移除"""
        if not self.remaining_points:
            return None, self.completed_count, self.total_original_count
        # 取出第一个
        self.current_target = self.remaining_points[0]
        self.current_target[2] = OBJECT_Z
        
        # 返回 (pose, 当前是第几个, 总数)
        # 注意：这里 idx 返回的是 "这是第几个被测的"，方便显示进度
        return self.current_target, self.completed_count + 1, self.total_original_count
        # if self.current_idx < self.total_count:
        #     pose = self.all_points[self.current_idx]
        #     self.current_idx += 1
        #     return pose, self.current_idx, self.total_count
        # return None, -1, self.total_count
    
    def mark_current_done(self):
        """确认当前点测试完成，保存进度"""
        if self.remaining_points:
            # 移除已完成的点 (就是列表第一个)
            self.remaining_points.pop(0)
            self.completed_count += 1
            self.save_progress()
    def save_progress(self):
        """将当前剩余点列表写入磁盘"""
        if not self.progress_file: return
        
        data = {
            "completed_count": self.completed_count,
            "remaining_points": self.remaining_points
        }
        
        #为了安全，先写临时文件再重命名
        temp_file = self.progress_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=4)
        os.replace(temp_file, self.progress_file)
        print(f"[Sampler] 进度已保存 ({len(self.remaining_points)} left)")
         
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

    def get_current_cartesian(self):
        # 辅助函数：获取当前绝对坐标 (XYZ+RPY)，单位转为米
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0: return None
        pose = np.array(pose, dtype=np.float32)
        pose[:3] /= 1000.0 # 这里的单位必须和训练时一致！如果训练除以1000，这里也要除
        return pose
    
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

    def recover_from_error(self, target_mode=0):
        """
        target_mode: 恢复后希望进入的模式。
        0: 位置模式 (用于 move_to)
        1: 伺服模式 (用于 set_servo_angle_j / 实时推理执行)
        """
        print(f"\n[Recovery] !!! 启动自动恢复程序，目标模式: {target_mode}")
        
        if self.arm is None: return

        # 1. 停止并清除错误 (这一步不分模式)
        self.arm.set_state(4)
        time.sleep(0.2)
        self.arm.clean_error()
        self.arm.clean_warn()
        time.sleep(0.2)

        # 2. 重新使能
        self.arm.motion_enable(enable=True)
        time.sleep(0.5)

        # 3. 关键：正确切换模式
        # 先 set_mode，再 set_state
        self.arm.set_mode(target_mode)
        time.sleep(0.2)
        self.arm.set_state(0)
        
        # 增加一个检查循环，确保模式切换成功后再退出函数
        # 这样可以避免退出函数后立刻执行 API 导致的 mode incorrect 警告
        for i in range(10):
            # 检查 SDK 缓存的模式是否已更新
            if self.arm.mode == target_mode:
                break
            time.sleep(0.1)
        
        if target_mode == 1:
            # 如果要回到伺服模式，先确保它在模式 0 下稍微往上抬一点，脱离碰撞点
            self.arm.set_mode(0)
            self.arm.set_state(0)
            curr_pos = self.arm.get_position()[1]
            curr_pos[2] += 20.0 # 向上抬 20mm
            self.arm.set_position(*curr_pos, wait=True)
            
            # 抬升完后再切换到推理所需的模式 1
            self.arm.set_mode(1)
            self.arm.set_state(0)

        print(f"[Recovery] 恢复完成，当前模式: {self.arm.mode}")
        
    def execute_action(self, action_delta):
        """
        执行单步动作 (Cartesian Delta Mode)
        action_delta: [dx, dy, dz, dRx, dRy, dRz, gripper] (单位: 米, 弧度)
        """
        # print(f"[Debug] Action Delta: {action_delta[:3]}")
        # 1. 获取当前绝对位姿 (米)
        curr_pose = self.get_current_cartesian()
        
        # 2. 计算目标绝对位姿 (Current + Delta)
        # 注意：这里做简单的欧拉角叠加。对于小步长控制通常足够。
        target_pose = curr_pose.copy()
        target_pose[:6] += action_delta[:6] 
        
        # 3. Z轴安全限位 (米 -> 米)
        # 注意 MIN_SAFE_Z 是毫米，这里要转成米比较，或者把 target 转回毫米
        target_z_mm = target_pose[2] * 1000.0
        if target_z_mm < MIN_SAFE_Z:
            # print(f"[Safety] Limit Z: {target_z_mm:.1f} -> {MIN_SAFE_Z}")
            target_pose[2] = MIN_SAFE_Z / 1000.0

        # 4. 准备 IK 输入 (米 -> 毫米)
        ik_target_pose = target_pose.copy()
        ik_target_pose[:3] *= 1000.0 # 转回 mm
        
        # 5. IK 解算
        # ret, target_joints = self.arm.get_inverse_kinematics(ik_target_pose, input_is_radian=True, return_is_radian=True)
        # 改进一下
        ret, target_joints, actual_target_pose = self.find_reachable_ik(curr_pose, target_pose)
        if ret == 0:
            # 6. 插值执行 (保持你原有的平滑逻辑)
            _, curr_joints_raw = self.arm.get_servo_angle(is_radian=True)
            
            
    
    
            curr_j = np.array(curr_joints_raw[:6])
            targ_j = np.array(target_joints[:6])
            
            diff = np.max(np.abs(np.array(curr_j[:6]) - np.array(target_joints[:6])))
            # print(f"[Debug] Max Joint Jump: {diff:.4f} rad")
            if diff > 6.28: # 如果一步跳变超过 0.5 弧度 (约30度)
                print("!!! DANGER: Joint jump too large! Stop!")
                return False # 拒绝执行
            
            duration = (1.0 / CONTROL_FREQ) * SLOW_DOWN_FACTOR
            steps = int(duration * INTERPOLATION_FREQ)
            if steps < 1: steps = 1
            
            for i in range(1, steps + 1):
                alpha = i / steps
                interp = curr_j + (targ_j - curr_j) * alpha
                # 【修改】增加返回值检查和自动恢复
                ret = self.arm.set_servo_angle_j(angles=np.append(interp, 0.0), is_radian=True)
                # 如果发送指令失败 (比如 code=9)
                if ret != 0:
                    print(f"[Hardware Error] set_servo_angle_j failed, code={ret}. Trying to recover...")
                    self.recover_from_error(target_mode=1) # 恢复后直接切回模式 1
                    return False # 这一步动作跳过
                
                time.sleep(1.0 / INTERPOLATION_FREQ)
        else:
            print("[Error] IK Failed. Target unreachable.")
            return False

        # 7. 夹爪
        target_gripper = action_delta[6]
        if target_gripper > 0.8: self.close_gripper()
        elif target_gripper < 0.2: self.open_gripper()
        
        return True
        
    def find_reachable_ik(self, start_pose, end_pose, search_steps=5):
        """
        如果在 end_pose IK 失败，则在 start 和 end 之间二分查找最近的可达点。
        """
        # 转换为 mm 以便 SDK 计算
        def get_ik(p):
            ik_p = p.copy()
            ik_p[:3] *= 1000.0
            return self.arm.get_inverse_kinematics(ik_p, input_is_radian=True, return_is_radian=True)

        # 1. 首先尝试原始目标
        ret, joints = get_ik(end_pose)
        if ret == 0:
            return ret, joints, end_pose

        # 2. 如果失败，尝试寻找“折中点”
        # 在当前位姿和目标位姿之间进行线性插值，从 0.8, 0.6, 0.4... 比例尝试
        print(f"[Warning] Original IK Failed. Searching for nearest reachable point...")
        
        # 尝试比例：0.75, 0.5, 0.25
        for ratio in [0.75, 0.5, 0.25, 0.1]:
            temp_pose = start_pose + (end_pose - start_pose) * ratio
            ret, joints = get_ik(temp_pose)
            if ret == 0:
                print(f"[Recovery] Found reachable point at {ratio*100:.0f}% of original step.")
                return ret, joints, temp_pose

        return -1, None, None
            
        
    # # 【还原】完全恢复 Code A 的执行逻辑，去掉所有额外检测
    # def execute_action(self, action_rad):
    #     # 1. 安全限位
    #     target_joints = [np.clip(a, l, h) for a, (l, h) in zip(action_rad[:6], JOINT_LIMITS)]
        
    #     # 2. Z轴检查与修正
    #     ret, pose = self.arm.get_forward_kinematics(angles=target_joints, input_is_radian=True, return_is_radian=True)
    #     if ret == 0:
    #         model_z = pose[2]
            
    #         # 只有当模型想去的高度 低于 安全限制时，才触发修正
    #         if model_z < MIN_SAFE_Z:
    #             print(f"[DEBUG] Z Limit Triggered! Model wants: {model_z:.2f}, Limit: {MIN_SAFE_Z}")
                
    #             # 构造修正后的位姿（保持XY和旋转不变，只把Z抬高到安全线）
    #             safe_pose = list(pose)
    #             safe_pose[2] = MIN_SAFE_Z
                
    #             # 重新解算关节角
    #             ret_ik, ik_joints = self.arm.get_inverse_kinematics(safe_pose, input_is_radian=True, return_is_radian=True)
    #             if ret_ik == 0: 
    #                 target_joints = list(ik_joints) # 使用修正后的关节角
    #             else: 
    #                 print("[Error] IK Failed during Z-safety adjustment")
    #                 return # IK 失败跳过该步骤

    #     # 3. 插值运动 (Time Dilation)
    #     code, current_joints = self.arm.get_servo_angle(is_radian=True)
    #     if code != 0: return

    #     curr_j = np.array(current_joints[:6])
    #     targ_j = np.array(target_joints[:6])
        
    #     duration = (1.0 / CONTROL_FREQ) * SLOW_DOWN_FACTOR
    #     steps = int(duration * INTERPOLATION_FREQ)
    #     if steps < 1: steps = 1
        
    #     # === 核心循环：这里绝对不能有任何 IO 阻塞 ===
    #     for i in range(1, steps + 1):
    #         alpha = i / steps
    #         interp = curr_j + (targ_j - curr_j) * alpha
    #         self.arm.set_servo_angle_j(angles=np.append(interp, 0.0), is_radian=True)
    #         time.sleep(1.0 / INTERPOLATION_FREQ)
    #     # ==========================================

    #     # 4. 夹爪
    #     g = action_rad[6]
    #     if g > 0.8: self.close_gripper()
    #     elif g < 0.2: self.open_gripper()

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
            self.open_gripper(); time.sleep(2.0)
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
        
    def is_in_boundary(self, pose_mm, boundary_points):
        """
        检查笛卡尔坐标(mm)是否在2D凸包范围内
        pose_mm: [x, y, z, ...]
        """
        # 提取 XY
        pt = (float(pose_mm[0]), float(pose_mm[1]))
        
        # 转换为 OpenCV 需要的 contour 格式 (int)
        # 注意：boundary_points_2d 是 float，这里为了 pointPolygonTest 最好保持精度
        # pointPolygonTest 支持 float 输入，但 contour 最好是 float32
        contour = boundary_points.astype(np.float32)
        
        # measureDist=False, 返回 +1(内), -1(外), 0(边)
        result = cv2.pointPolygonTest(contour, pt, False)
        return result >= 0

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
def main():
    print(f"Loading Model: {CONFIG_NAME}...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    robot = XArmHardware(ROBOT_IP, CAMERAS)
    sampler = TaskSampler(POINTS_FILE, progress_file=PROGRESS_FILE, resume=RESUME_TESTING)
    viz = TaskVisualizer(VIS_SAVE_DIR, RESULT_IMG_NAME, BOUNDARY_POINTS_2D, HOME_POS)
    debugger = DebugVisualizer(MIN_SAFE_Z, VIS_SAVE_DIR)
    recorder = MetricsRecorder()

    # 启动后台键盘监听
    kb = KeyboardThread()
    kb.start()
    # 暂停标志位初始化
    pause_requested = False 
    
    # prompt = "pick up the industrial components B"
    current_target = None
    
    # 【移除了】 start_pose_abs = robot.get_current_cartesian() 
    # 原因：放在这里会导致第二轮抓取时基准点失效

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
                # 可选：测试完成后删除进度文件，方便下次重来
                # if os.path.exists(PROGRESS_FILE): os.remove(PROGRESS_FILE)
                break
            print(f"\n=== Test Point {idx}/{total} ===")
            print(f"Target: {target_pose}")
            
            # 2. 机器人去目标点放置物体 (Setup)
            robot.run_setup(target_pose)
            print(">>> Setup Done. Starting Inference...")            
            
            # 3. 推理准备
            robot.flush_cameras() 
            
            # 【新增位置】在这里获取本回合的起始基准点
            start_pose_abs = robot.get_current_cartesian()
            
            # 开始记录：传入 起始点 和 理论目标点
            recorder.start_episode(start_pose_abs, target_pose)
            
            robot.arm.set_mode(1)
            robot.arm.set_state(0)
            time.sleep(0.5) # 等待模式切换生效
            #   初始化标志位
            just_recovered = False 
            
            # 【删除了】原先这里的 policy.infer 和 robot.move_to_start
            # 原因：模型输出的是相对增量，不能直接用于 move_to_start 的绝对位置控制
            
            # 4. AI 控制循环
            print(">>> AI Loop running... Press 'o' to ABORT (Mark as Fail).")
            aborted = False
            # 第1次运行需要编译模型(JIT)，给 120秒，后续给 27秒
            current_timeout_limit = 120.0 if episode == 1 else 27.0
            # 计时器初始化
            episode_start_time = time.time()
            consecutive_re_inference_count = 0
            MAX_RETRY = 10 # 增加重试次数，因为现在有回拉机制，更容易救回来
            close_gripper_time = time.time()
            
            if robot.arm.mode != 1:
                print(">>> robot.arm.mode != 1 , enter recover_from_error()")
                robot.recover_from_error(target_mode=1)
            while True:
                # 【新增】重置本轮开始时间
                #  极速检查退出，不要用 select 阻塞
                if kb.get_and_clear_key() == 'o':
                    aborted = True; break
                elif kb.get_and_clear_key() == 'p':
                    if not pause_requested:
                        print("\n>>> ⏳ [指令收到] 本轮结束后将暂停...")
                        pause_requested = True
                elif kb.get_and_clear_key() == 'y':
                    print("\n>>> 🎯 [指令收到] 手动触发抓取 (Mark as Success).")
                    # 立即闭合夹爪 (模拟模型输出了闭合)
                    robot.close_gripper()
                    time.sleep(0.5) # 给一点时间闭合
                    # 标记为成功退出 (不设aborted)
                    # 记录夹爪闭合时间 (用于计算耗时)
                    close_gripper_time = time.time()
                    # 跳出推理循环，直接进结算
                    break 
                
                # 1. 观测 (Code A: get_observation)
                raw_obs = robot.get_observation()
                # =================================================================
                # 【DEBUG】如果是回拉后刚回来的第一帧，立刻保存，看看到底喂给了模型什么
                # =================================================================
                if just_recovered:
                    debug_rec_dir = os.path.join(VIS_SAVE_DIR, "debug_recovery_check")
                    os.makedirs(debug_rec_dir, exist_ok=True)
                    timestamp = int(time.time() * 1000)
                    save_path = os.path.join(debug_rec_dir, f"recovery_input_{timestamp}.jpg")
                    
                    print(f"\n[DEBUG CHECK] 正在保存回拉后的首帧推理图像: {save_path}")
                    if 'cam_left_wrist' in raw_obs:
                        # 注意：raw_obs 是 RGB，保存需转 BGR
                        cv2.imwrite(save_path, cv2.cvtColor(raw_obs['cam_left_wrist'], cv2.COLOR_RGB2BGR))
                    
                    # 存完后重置标志位，只存这一张
                    just_recovered = False
                # =================================================================
                curr_pose_abs = robot.get_current_cartesian() # 当前绝对坐标
                
                # 构造相对输入 State
                # State = 当前绝对 - 起始绝对
                def normalize_angle(angle):
                    # 将角度映射到 -pi 到 pi
                    return (angle + np.pi) % (2 * np.pi) - np.pi
                rel_pose = curr_pose_abs - start_pose_abs
                rel_pose[5] = normalize_angle(curr_pose_abs[5] - start_pose_abs[5])
                print(f"\r[State] Rel_pose: {rel_pose[:3]}", end="") 
                
                # 拼装 (7维)
                input_state = np.append(rel_pose, robot.current_gripper_state)
                
                # 2. 推理
                result = policy.infer({
                    "cam_left_wrist": raw_obs["cam_left_wrist"],
                    "cam_right_wrist": raw_obs["cam_right_wrist"],
                    "state": input_state, "prompt": TASK_PROMOT
                })
                
                # 模型输出的是 Delta Action Chunk [T, 7]
                action_chunk = np.array(result["actions"])
                
                debugger.update(raw_obs, action_chunk, robot.arm)
                
                # 3. 抓取检测
                if np.any(action_chunk[:1, 6] > 0.8):
                    close_gripper_time = time.time() # 【关键】自动抓取也要记录时间
                    print(">>> Auto Grasp Detected.")
                    break
                
                # 4. 执行 (完全一致的循环结构)
                steps_to_run = min(EXECUTE_STEPS, len(action_chunk))
                # for i in range(steps_to_run):
                #     step_start = time.time()
                    
                #     # 再次极速检查停止
                #     if kb.get_and_clear_key() == 'o': 
                #         aborted = True; break
                    
                #     # 执行动作 (注意：execute_action 必须是你修改过的支持 Delta 的版本)
                #     robot.execute_action(action_chunk[i])
                    
                #     # 频率控制 (Code A 逻辑)
                #     elapsed = time.time() - step_start
                #     sleep_time = (1.0 / CONTROL_FREQ) - elapsed
                #     if sleep_time > 0: time.sleep(sleep_time)
                need_re_inference = False
                for i in range(steps_to_run):
                    current_duration = time.time() - episode_start_time
                    if current_duration > episode_start_time:
                        print(f"\n[Timeout] 耗时 {current_duration:.1f}s > 27s. 强制中断 Chunk，重新推理...")
                        aborted = True
                        break # 跳出 for 循环 -> 进入下一次 while True (重新拍照推理)

                    raw_action = action_chunk[i]
                    
                    # 【修改逻辑】：如果模型输出的是 "相对于Start的位置"
                    # Target = Start_Pose + Model_Output
                    # 我们需要算出它相对于 Current 的 Delta 传给 execute_action
                    
                    # 预测的目标绝对位置
                    pred_target_abs = start_pose_abs[:6] + raw_action[:6]
                    
                    # 转换成 mm 进行检测
                    # =======================================================
                    pred_target_mm = pred_target_abs * 1000.0
                    
                    # 使用宽松边界 (EXPANDED) 进行检查
                    if not robot.is_in_boundary(pred_target_mm, BOUNDARY_EXPANDED):
                        print(f"\n[Safety] 目标 ({pred_target_mm[0]:.0f}, {pred_target_mm[1]:.0f}) 超出宽松边界！正在回拉...")
                        # =======================================================
                        # 【DEBUG 新增】: 保存当前帧及后续缓冲帧，验证是否有延迟
                        # =======================================================
                        debug_dir = os.path.join(VIS_SAVE_DIR, "debug_pullback")
                        os.makedirs(debug_dir, exist_ok=True)
                        timestamp = int(time.time() * 1000)
                        
                        print(f"[Debug] 正在保存异常时刻图像到: {debug_dir}")
                        # 1. 保存导致这次错误推理的“案发现场”图片 (raw_obs)
                        # 注意：raw_obs 是 RGB，OpenCV 保存需要转 BGR
                        if 'cam_left_wrist' in raw_obs:
                            cv2.imwrite(
                                os.path.join(debug_dir, f"{timestamp}_00_inference_input.jpg"), 
                                cv2.cvtColor(raw_obs['cam_left_wrist'], cv2.COLOR_RGB2BGR)
                            )
                        
                        # 2. 连续读取并保存接下来的 5 帧，看看缓冲区里是什么
                        # 如果这 5 张图变化巨大，或者第 1 张和第 5 张位置差异很大，说明缓冲区有严重积压
                        for i in range(1, 6):
                            temp_obs = robot.get_observation() # 这里面包含了一次 read()
                            if 'cam_left_wrist' in temp_obs:
                                cv2.imwrite(
                                    os.path.join(debug_dir, f"{timestamp}_{i:02d}_buffer_flush.jpg"), 
                                    cv2.cvtColor(temp_obs['cam_left_wrist'], cv2.COLOR_RGB2BGR)
                                )
                            # 稍微 sleep 一点点，模拟处理时间，或者全速读以测试纯 I/O 堆积
                            time.sleep(0.01) 
                        # =======================================================
                        
                        # --- 计算回拉向量 ---
                        # 策略：向区域中心点回拉
                        center_pt = np.mean(BOUNDARY_POINTS_2D, axis=0) # 原始严格边界的中心
                        curr_xy = pred_target_mm[:2]
                        
                        # 计算方向向量: Current -> Center
                        vec_to_center = center_pt - curr_xy
                        norm = np.linalg.norm(vec_to_center)
                        
                        if norm > 0:
                            # 归一化并乘以回拉距离 (例如 50mm)
                            # pull_back_vec = (vec_to_center / norm) * 50.0
                            pull_back_vec = (vec_to_center / norm) * 50.0
                        else:
                            pull_back_vec = np.array([10.0, 10.0]) # 异常保护
                            
                        # --- 执行回拉动作 ---
                        # 获取当前位置
                        curr_pose_recover = robot.get_current_cartesian()
                        # 目标位置 = 当前位置 + 回拉向量 (注意单位换算 mm -> m)
                        target_pose_recover = curr_pose_recover.copy()
                        target_pose_recover[0] += pull_back_vec[0] / 1000.0
                        target_pose_recover[1] += pull_back_vec[1] / 1000.0
                        # Z轴保持不变或稍微抬高一点点防止拖拽
                        # target_pose_recover[2] += 0.01 
                        
                        # 计算 Delta 并执行
                        delta_recover = target_pose_recover - curr_pose_recover
                        # 拼装 Action (夹爪保持当前状态)
                        action_recover = np.append(delta_recover[:6], robot.current_gripper_state)
                        
                        print(f"[Recovery] 执行回拉动作: dX={pull_back_vec[0]:.1f}mm, dY={pull_back_vec[1]:.1f}mm")
                        robot.execute_action(action_recover)
                        # =======================================================
                        # 回拉后：清理相机缓存 + 原地停留 2 秒，再开始下一次推理
                        # 目的：尽量模拟 episode 开始时的 "flush + 稳定一下" 的观测条件
                        # =======================================================
                        try:
                            print("[Recovery] 回拉完成：清理相机缓存，并原地等待 2 秒后重新推理...")
                            robot.flush_cameras()
                            time.sleep(2.0)
                            robot.flush_cameras()
                        except Exception as e:
                            print(f"[Recovery] 相机 flush/等待过程中出现异常（忽略继续）：{e}")
                        
                        # =======================================================
                        # 【关键修复】回拉后重置 start_pose_abs 为当前回拉后的位置
                        # 原因：模型输出的 action 是"相对 start_pose_abs 的目标位置"
                        # 如果回拉后不重置，坐标系会错乱，导致朝错误方向运动
                        # =======================================================
                        new_start_pose = robot.get_current_cartesian()
                        if new_start_pose is not None:
                            print(f"[Recovery] 重置 episode 起点：从 {np.round(start_pose_abs[:3]*1000, 1)} 更新为 {np.round(new_start_pose[:3]*1000, 1)} (mm)")
                            start_pose_abs = new_start_pose
                        else:
                            print("[Recovery] ⚠️ 警告：无法获取回拉后位置，start_pose_abs 未更新")
                        
                        # 强制触发重推理
                        need_re_inference = True
                        # 标记：刚才发生了回拉，下一次循环开头要查图
                        just_recovered = True 
                        break 
                    # =======================================================
                
                    # 当前绝对位置
                    curr_pose_abs = robot.get_current_cartesian()
                    
                    # 计算需要的 Delta
                    real_delta = pred_target_abs - curr_pose_abs
                    
                    # 拼装夹爪
                    action_to_execute = np.append(real_delta, raw_action[6])
                    
                    success = robot.execute_action(action_to_execute) # execute_action 内部会限幅
                    if not success:
                        print("\n[Safety] 动作执行失败 (关节跳变/IK)。启动【主动恢复】策略...")
                        
                        # --- 1. 计算恢复向量 (向中心回拉 + 向上抬起) ---
                        # 获取当前位置 (米)
                        curr_pose_m = robot.get_current_cartesian()
                        curr_xy_mm = curr_pose_m[:2] * 1000.0
                        
                        # 计算中心点 (mm)
                        center_pt = np.mean(BOUNDARY_POINTS_2D, axis=0)
                        
                        # 计算指向中心的方向
                        vec_to_center = center_pt - curr_xy_mm
                        dist_to_center = np.linalg.norm(vec_to_center)
                        
                        # 构造恢复动作 Delta (单位: 米)
                        # 格式: [dx, dy, dz, dr, dp, dy, gripper]
                        recovery_delta = np.zeros(7)
                        
                        # A. XY平面的回拉 (回拉 30mm)
                        if dist_to_center > 0:
                            direction = vec_to_center / dist_to_center
                            # 往中心拉 0.03米
                            recovery_delta[0] = direction[0] * 0.03 
                            recovery_delta[1] = direction[1] * 0.03
                        
                        # B. Z轴的抬升 (抬起 20mm) - 这一步对解决关节跳变非常有效
                        recovery_delta[2] = 0.02 
                        
                        # C. 保持夹爪状态不变
                        recovery_delta[6] = robot.current_gripper_state
                        
                        print(f"[Recovery] 执行避险动作: 向中心回拉 3cm, 向上抬起 2cm...")
                        
                        # --- 2. 执行恢复动作 ---
                        # 再次调用 execute_action 执行这个人工生成的动作
                        # 如果这次还失败，那就没办法了，只能交给外层的 MAX_RETRY 去处理
                        rec_success = robot.execute_action(recovery_delta)
                        
                        if rec_success:
                            print("[Recovery] 避险动作执行成功。重新开始推理。")
                            # =======================================================
                            # 【关键修复】避险动作后也重置 start_pose_abs
                            # 原因：机械臂位置已改变，需要更新坐标系原点
                            # =======================================================
                            new_start_pose = robot.get_current_cartesian()
                            if new_start_pose is not None:
                                print(f"[Recovery] 重置 episode 起点：从 {start_pose_abs[:3]*1000:.1f} 更新为 {new_start_pose[:3]*1000:.1f} (mm)")
                                start_pose_abs = new_start_pose
                        else:
                            print("[Recovery] 避险动作也失败了 (可能卡死)。")
                        
                        # 无论避险是否成功，都必须中断当前 Chunk，重新拍照推理
                        need_re_inference = True
                        break
                    else:
                        close_gripper_time = time.time()
                
                    key = kb.get_and_clear_key()
                    if key == 'o': 
                        aborted = True
                        break
                    elif key == 'p':
                        if not pause_requested:
                            print("\n>>> ⏳ [指令收到] 本轮结束后将暂停...")
                            pause_requested = True
                    elif key == 'y':
                        print("\n>>> 🎯 [指令收到] 手动触发抓取 (Mark as Success).")
                        robot.close_gripper()
                        close_gripper_time = time.time()
                        aborted = False
                        # 这里需要双重 break，先设置标志位
                        break 
                    # 记录每一步的实际位置
                    # 确保 execute_action 之后机械臂已经动了
                    curr_pos_abs = robot.get_current_cartesian()
                    recorder.step(curr_pos_abs)
                
                if aborted: break
                if key == 'y':
                    break
                # 如果是因为触发了上述三个保护机制而 break 的
                if need_re_inference:
                    consecutive_re_inference_count += 1
                    if consecutive_re_inference_count >= MAX_RETRY:
                        print(f"\n[Failure] 连续 {MAX_RETRY} 次重推理/回拉无效。判定失败。")
                        aborted = True
                        break
                    continue
                else:
                    consecutive_re_inference_count = 0
            
            # 5. 结算环节
            if aborted:
                print("\n>>> Inference Aborted by 'o'. Marking as FAILURE.")
                viz.update_result(target_pose, False) # 记为失败
                robot.open_gripper()
                target_up = list(target_pose); target_up[2] += 100
                robot.arm.set_mode(0); robot.arm.set_state(0)
                robot.arm.set_position(*target_up, speed=300, wait=True, is_radian=True)
                target_pose_withObjetcZ = list(target_pose);target_up[2] = OBJECT_Z
                robot.arm.set_position(*target_pose_withObjetcZ, speed=300, wait=True, is_radian=True)
                robot.close_gripper(); time.sleep(2.0)
                robot.arm.set_position(*target_up, speed=300, wait=True, is_radian=True)
                
                current_metrics = recorder.end_episode(success=False, close_gripper_time = close_gripper_time)
                robot.arm.set_position(*HOME_POS, speed=300, wait=True, is_radian=True)

            else:
                # 正常结束，等待人工判定
                robot.close_gripper()
                time.sleep(2.0)
                current_metrics = recorder.end_episode(success=True, close_gripper_time = close_gripper_time)
                robot.move_home_scripted()
                print(">>> Marked as SUCCESS.")
                viz.update_result(target_pose, True)
                
                print("\n>>> Evaluate Result: [y] Success / [n] Failure")
                # 循环等待直到按下 y 或 n
                # while True:
                #     k = kb.get_and_clear_key()
                    
                #     if k == 'y': 
                #         print(">>> Marked as SUCCESS.")
                #         current_metrics = recorder.end_episode(success=True, close_gripper_time = close_gripper_time)
                #         robot.arm.set_mode(0)
                #         robot.arm.set_state(0)
                #         time.sleep(0.1) # 等待固件切换完成
                #         target_up = list(target_pose); target_up[2] += 100
                #         robot.arm.set_position(*target_up, speed=100, wait=True, is_radian=True)
                #         robot.move_home_scripted()
                #         viz.update_result(target_pose, True)
                #         break
                #     elif k == 'n': 
                #         print(">>> Marked as FAILURE.")
                #         current_metrics = recorder.end_episode(success=False, close_gripper_time = close_gripper_time)
                #         robot.arm.set_mode(0)
                #         robot.arm.set_state(0)
                #         time.sleep(0.1) # 等待固件切换完成
                #         robot.open_gripper(); time.sleep(1.0)
                #         target_up = list(target_pose); target_up[2] += 100

                #         robot.arm.set_position(*target_up, speed=300, wait=True, is_radian=True)
                #         robot.arm.set_position(*target_pose, speed=300, wait=True, is_radian=True)
                #         robot.close_gripper(); time.sleep(2.0)
                #         robot.arm.set_position(*target_up, speed=300, wait=True, is_radian=True)
                
                #         robot.arm.set_position(*target_up, speed=100, wait=True, is_radian=True)
                #         robot.move_home_scripted()
                #         viz.update_result(target_pose, False)
                #         break
                #     time.sleep(0.05)
            
            # 【新增】关键步骤：测试结束（无论成功失败），标记为完成并保存进度
            sampler.mark_current_done()   
            recorder.print_current_metrics(current_metrics)
            recorder.print_summary()
            
            if pause_requested:
                print("\n" + "="*50)
                print(">>> ⏸️  程序已暂停 (用户请求)。")
                print(">>> ⌨️  请按 [ENTER] 键继续下一轮测试...")
                print("="*50)
                
                # 循环等待回车
                while True:
                    k = kb.get_and_clear_key()
                    if k == '\n' or k == '\r': # 回车键
                        print(">>> ▶️  继续运行...")
                        pause_requested = False # 重置标志位
                        break
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopped.")
        recorder.print_summary()
    finally:
        kb.stop()
        robot.close()

if __name__ == "__main__":
    main()