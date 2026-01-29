#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marvin机械臂模型推理代码
基于xarm_infer.py适配，使用Marvin机械臂API
"""

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
import json

# OpenPI 依赖
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
openpi_client_path = os.path.join(current_dir, "../../../packages/openpi-client/src")
sys.path.append("/home/openpi/src")
sys.path.append(os.path.abspath(openpi_client_path))
sys.path.insert(0, parent_dir)

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi_client import image_tools

# Marvin SDK
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# 将该目录加入 Python 搜索路径
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from SDK_PYTHON.fx_kine import Marvin_Kine, FX_InvKineSolvePara
from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS

import matplotlib
matplotlib.use('Agg') # 设置后端为非交互式，Docker 专用
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. 配置区域
# -----------------------------------------------------------------------------
ROBOT_IP = "10.10.13.12"
CONFIG_NAME = "pi05_xarm_1212_night"
CHECKPOINT_DIR = "/home/openpi/checkpoints/exp1/1000"
VIS_SAVE_DIR = "/home/openpi/examples/xarm_real/images"
TASK_PROMOT = "pick up the object"
SELECTED_TASK = "C"  

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
        # "prompt": "pick up the small upright valve",
        "prompt": "pick up the object",
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
TASK_PROMOT = current_config["prompt"]

# 基础坐标 (X, Y, Roll, Pitch, Yaw) - 单位：毫米，度
_base_x = 554.626923
_base_y = 361.343277
_base_r = 3.12897
_base_p = 0.012689
_base_yw = -1.01436

# POS_A 将在 main() 中设置为与 HOME_POS 相同的值
# 注意：将在 main() 中通过正运动学从 HOME_JOINTS 计算得出
POS_A = None  # 占位符，实际值将在 main() 中计算

# 自动生成结果图片文件名
RESULT_IMG_NAME = f"0127afternoon_cloudyDay_exp23_24000_test1_components_{SELECTED_TASK}.png"
print(f">>> 当前任务: [{SELECTED_TASK}]")
print(f">>> Prompt: {TASK_PROMOT}")
# POS_A 将在 main() 中设置，这里不打印

# 指定要读取的点位文件
POINTS_FILE = os.path.join(VIS_SAVE_DIR, "test_points.json")
RESUME_TESTING = True
PROGRESS_FILE = os.path.join(VIS_SAVE_DIR, "test_progress.json")

CAMERAS = {
    # "cam_left_wrist": "/dev/cam_left_wrist",
    "cam_right_wrist": 2
}

CROP_CONFIGS = {
    # "cam_left_wrist": (118, 60, 357, 420),
    # "cam_right_wrist": (136, 57, 349, 412)
}

CONTROL_FREQ = 10 
EXECUTE_STEPS = 2
MIN_SAFE_Z = -119  # 单位：毫米
SLOW_DOWN_FACTOR = 2.0  
INTERPOLATION_FREQ = 100.0 

# Home点：关节角度（度）- 从record_fixed_path_marvin.py获取
HOME_JOINTS = [83.20, 31.53, -22.34, 58.78, -79.30, 13.48, 88.53]
# 初始状态：关节角全0
INIT_JOINTS = [86.69, 27.42, -21.83, 58.79, -84.21, 13.56, 88.55]

# exp9 boundary
BOUNDARY_POINTS_2D = np.array([
    [-1000.6, 1000.5],
    [-1000.0, -1000.2],
    [1000.9, -1000.4],
    [1000.1, 1000.0],
])

# 计算宽松边界
_center_point = np.mean(BOUNDARY_POINTS_2D, axis=0)
BOUNDARY_EXPANDED = _center_point + (BOUNDARY_POINTS_2D - _center_point) * 100.15

# Home点笛卡尔位姿（用于可视化）- 单位：毫米，度
# 注意：将在 main() 中通过正运动学从 HOME_JOINTS 计算得出
HOME_POS = None  # 占位符，实际值将在 main() 中计算

# 这些变量将在 main() 中从 POS_A 设置
# 临时设置默认值，避免模块加载时出错
FIXED_Z = current_config["z"]  # 临时值，将在 main() 中更新
FIXED_ROLL = _base_r
FIXED_PITCH = _base_p
BASE_YAW = _base_yw
YAW_RANDOM_RANGE = (-np.pi/6, np.pi/6)

# Marvin配置
ARM_NAME = 'A'
ARM_TYPE = 0
CONFIG_FILE = os.path.join(root_dir, 'SDK_PYTHON', 'ccs_m6_40.MvKDCfg')

# 运动速度（百分比）
VEL_RATIO_FAST = 30
VEL_RATIO_SERVO = 10
ACC_RATIO = 10

# 夹爪配置
GRIPPER_COM_PORT = 2
GRIPPER_SLAVE_ID = 9


# -----------------------------------------------------------------------------
# 2. 工具类（保持与xarm_infer.py一致）
# -----------------------------------------------------------------------------
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
            "start_pose": np.array(start_pose_abs) * 1000.0, 
            "ground_truth_pose": np.array(ground_truth_pose),
            "trajectory": [], 
            "success": False,
            "steps": 0
        }
        self.current_episode["trajectory"].append(self.current_episode["start_pose"][:3])
        self.current_episode["final_pos_mm"] = self.current_episode["start_pose"][:3]
        self.current_episode["final_rpy_rad"] = start_pose_abs[3:]
        
    def step(self, current_pose_abs):
        """记录每一步的实际位置 (输入单位: 米)"""
        pos_mm = np.array(current_pose_abs[:3]) * 1000.0
        self.current_episode["trajectory"].append(pos_mm)
        self.current_episode["steps"] += 1
        self.current_episode["final_full_pose"] = np.array(current_pose_abs) * 1000.0
        self.current_episode["final_pos_mm"] = pos_mm
        self.current_episode["final_rpy_rad"] = current_pose_abs[3:]

    def end_episode(self, success, close_gripper_time):
        self.current_episode["end_time"] = close_gripper_time
        self.current_episode["success"] = success
        metrics = self._calculate_single_metrics(self.current_episode)
        self.episode_metrics.append(metrics)
        return metrics

    def _calculate_single_metrics(self, data):
        duration = data["end_time"] - data["start_time"]
        
        traj = np.array(data["trajectory"])
        if len(traj) > 3:
            vel = np.diff(traj, axis=0)
            acc = np.diff(vel, axis=0)
            jerk = np.diff(acc, axis=0)
            avg_jerk = np.mean(np.linalg.norm(jerk, axis=1))
        else:
            avg_jerk = 0.0

        gt_pos = data["ground_truth_pose"][:3]
        start_pos = data["start_pose"][:3]
        dist_xy = np.linalg.norm(gt_pos[:2] - start_pos[:2])
        dist_z = abs(gt_pos[2] - start_pos[2])
        ideal_path_len = dist_xy + dist_z
        
        REF_STEP_LEN_MM = 5.0
        opt_steps = max(1, int(ideal_path_len / REF_STEP_LEN_MM))
        step_ratio = data["steps"] / opt_steps

        final_pos = data["final_pos_mm"]
        pos_error = np.linalg.norm(final_pos - gt_pos)
        
        gt_rpy = data["ground_truth_pose"][3:]
        final_rpy = data["final_rpy_rad"]
        diff_rpy_deg = np.degrees(np.abs(final_rpy - gt_rpy))
        rot_error = np.sum(diff_rpy_deg)

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
        
        if num_success > 0:
            avg_time = np.mean([m["time"] for m in success_list])
            avg_jerk = np.mean([m["jerk"] for m in success_list])
            avg_step_ratio = np.mean([m["step_ratio"] for m in success_list])
            avg_pos_error = np.mean([m["pos_error"] for m in success_list])
            avg_rot_error = np.mean([m["rot_error"] for m in success_list])
        else:
            avg_time = 0.0
            avg_jerk = 0.0
            avg_step_ratio = 0.0
            avg_pos_error = 0.0
            avg_rot_error = 0.0

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
        status_str = "✅ Success" if metrics["success"] == 1.0 else "❌ Failure"
        print("-" * 40)
        print(f"📝 本轮详细数据 ({status_str})")
        print(f"   ⏱️ 耗时:       {metrics['time']:.2f} s")
        print(f"   🎯 位置误差:   {metrics['pos_error']:.2f} mm")
        print(f"   📐 角度误差:   {metrics['rot_error']:.2f} deg")
        print(f"   👣 步数比:     {metrics['step_ratio']:.2f} (Opt: {metrics['opt_steps']:.0f})")
        print(f"   📉 平滑度:     {metrics['jerk']:.4f}")
        print("-" * 40)


class DebugVisualizer:
    def __init__(self, safe_z_limit, save_dir):
        self.fig, self.axs = plt.subplots(3, 2, figsize=(10, 12))
        self.safe_z_limit = safe_z_limit
        self.save_path = os.path.join(save_dir, "live_debug_status.png")
        
        self.ax_cam1 = self.axs[0, 0]
        self.ax_cam2 = self.axs[0, 1]
        self.ax_xy_plane = self.axs[1, 0]
        self.ax_z = self.axs[1, 1]
        self.ax_xy_time = self.axs[2, 0]
        self.ax_grip = self.axs[2, 1]
        
        self.ax_xy_plane.set_title("XY Trajectory (Top-Down View)")
        self.ax_xy_plane.set_xlabel("X (mm)")
        self.ax_xy_plane.set_ylabel("Y (mm)")
        self.ax_xy_plane.grid(True)
        self.ax_xy_plane.set_aspect('equal', adjustable='box')
        
        self.ax_z.set_title("Z Trajectory (Height)")
        self.ax_z.set_ylabel("Z (mm)")
        self.ax_z.axhline(y=safe_z_limit, color='r', linestyle='--', label='Limit')
        self.ax_z.grid(True)
        
        self.ax_xy_time.set_title("X & Y over Time (Steps)")
        self.ax_xy_time.set_ylabel("Position (mm)")
        self.ax_xy_time.grid(True)
        
        self.ax_grip.set_title("Gripper Intent")
        self.ax_grip.set_ylim(-0.1, 1.1)
        self.ax_grip.axhline(y=0.8, color='g', linestyle='--', label='Trigger')
        self.ax_grip.grid(True)

        print(f"[Vis] Debug visualization will be saved to: {self.save_path}")

    def _clear_lines(self, ax):
        for line in list(ax.lines):
            line.remove()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    def update(self, obs, action_chunk, robot_hw):
        self.ax_cam1.clear(); self.ax_cam1.set_title("Left Wrist")
        self.ax_cam1.imshow(obs['cam_left_wrist'])
        self.ax_cam1.axis('off')
        
        self.ax_cam2.clear(); self.ax_cam2.set_title("Right Wrist")
        self.ax_cam2.imshow(obs['cam_right_wrist'])
        self.ax_cam2.axis('off')
        
        curr_pose = robot_hw.get_current_cartesian()
        if curr_pose is None: return
        
        curr_x, curr_y, curr_z = curr_pose[0], curr_pose[1], curr_pose[2]
        pred_x, pred_y, pred_z = [], [], []
        start_x, start_y, start_z = curr_x, curr_y, curr_z
        sim_x, sim_y, sim_z = curr_x, curr_y, curr_z
        
        for i in range(len(action_chunk)):
            dx = action_chunk[i][0] * 1000.0
            dy = action_chunk[i][1] * 1000.0
            dz = action_chunk[i][2] * 1000.0
            
            cur_pred_x = start_x + dx
            cur_pred_y = start_y + dy
            cur_pred_z = start_z + dz
            
            pred_x.append(cur_pred_x)
            pred_y.append(cur_pred_y)
            pred_z.append(cur_pred_z)
        
        steps = np.arange(len(pred_x))

        self._clear_lines(self.ax_xy_plane)
        self.ax_xy_plane.plot(pred_x, pred_y, 'b-o', alpha=0.6, markersize=4, label='Path')
        if pred_x:
            self.ax_xy_plane.plot(pred_x[0], pred_y[0], 'go', markersize=8, label='Start')
            self.ax_xy_plane.plot(pred_x[-1], pred_y[-1], 'rx', markersize=8, label='End')
            
            mid_x, span_x = (np.min(pred_x) + np.max(pred_x))/2, (np.max(pred_x) - np.min(pred_x))
            mid_y, span_y = (np.min(pred_y) + np.max(pred_y))/2, (np.max(pred_y) - np.min(pred_y))
            max_span = max(span_x, span_y, 20) 
            self.ax_xy_plane.set_xlim(mid_x - max_span, mid_x + max_span)
            self.ax_xy_plane.set_ylim(mid_y - max_span, mid_y + max_span)
            
        self.ax_xy_plane.legend(loc='upper right', fontsize='small')

        self._clear_lines(self.ax_z)
        self.ax_z.axhline(y=self.safe_z_limit, color='r', linestyle='--')
        self.ax_z.plot(steps, pred_z, 'b-o', markersize=4)
        if pred_z:
            min_z = min(min(pred_z), self.safe_z_limit)
            self.ax_z.set_ylim(min_z - 20, max(pred_z) + 20)

        self._clear_lines(self.ax_xy_time)
        self.ax_xy_time.plot(steps, pred_x, 'c--', label='X')
        self.ax_xy_time.plot(steps, pred_y, 'm--', label='Y')
        self.ax_xy_time.legend(loc='best', fontsize='small')

        grip_vals = action_chunk[:, 6]
        self._clear_lines(self.ax_grip)
        self.ax_grip.axhline(y=0.8, color='g', linestyle='--')
        self.ax_grip.plot(steps, grip_vals, 'k-x')

        try:
            self.fig.canvas.draw()
            img_rgba = np.asarray(self.fig.canvas.buffer_rgba())
            image = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(self.save_path, image)
        except Exception as e:
            print(f"[Vis Error] {e}")


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
        print(f"[Debug] Saving image to: {save_path_str}")
        try:
            cv2.imwrite(save_path_str, self.canvas)
            print(f"[Vis] Result saved successfully.")
        except Exception as e:
            print(f"[Error] Failed to save image: {e}")


class TaskSampler:
    def __init__(self, json_path, progress_file=None, resume=False):
        self.original_json_path = json_path
        self.progress_file = progress_file
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到点位文件: {json_path}。请先运行生成脚本。")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.all_points_original = data["grid"] + data["boundary"]
            
        if resume and progress_file and os.path.exists(progress_file):
            print(f"[Sampler] 发现进度文件: {progress_file}，尝试恢复...")
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                self.remaining_points = progress_data.get("remaining_points", [])
                self.completed_count = progress_data.get("completed_count", 0)
                
                if not self.remaining_points and self.completed_count > 0:
                    print("[Sampler] ⚠️ 进度文件显示所有点已测试完毕！")
                else:
                    print(f"[Sampler] ✅ 成功恢复进度。已测: {self.completed_count}, 剩余: {len(self.remaining_points)}")
            except Exception as e:
                print(f"[Sampler] ❌ 读取进度文件失败 ({e})，将重置为全部测试点。")
                self.remaining_points = list(self.all_points_original)
                self.completed_count = 0
        else:
            print("[Sampler] 初始化新测试序列...")
            self.remaining_points = list(self.all_points_original)
            self.completed_count = 0
            
            if resume and progress_file:
                self.save_progress()
        self.total_original_count = len(self.all_points_original)
        self.current_target = None
        self.current_idx = 0
        
        print(f"[Sampler] Loaded {self.total_original_count} points (Grid + Boundary).")
    
    def get_next_target(self):
        if not self.remaining_points:
            return None, self.completed_count, self.total_original_count
        self.current_target = self.remaining_points[0]
        self.current_target[2] = OBJECT_Z
        return self.current_target, self.completed_count + 1, self.total_original_count
    
    def mark_current_done(self):
        if self.remaining_points:
            self.remaining_points.pop(0)
            self.completed_count += 1
            self.save_progress()
            
    def save_progress(self):
        if not self.progress_file: return
        data = {
            "completed_count": self.completed_count,
            "remaining_points": self.remaining_points
        }
        temp_file = self.progress_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=4)
        os.replace(temp_file, self.progress_file)
        print(f"[Sampler] 进度已保存 ({len(self.remaining_points)} left)")


# -----------------------------------------------------------------------------
# 3. Marvin硬件封装
# -----------------------------------------------------------------------------
class MarvinHardware:
    def __init__(self, ip, camera_indices, config_file):
        print(f"Connecting to MARVIN robot at {ip}...")
        
        self.dcss = DCSS()
        self.arm = Marvin_Robot()
        self.kine = Marvin_Kine()
        self.kine.log_switch(0)
        
        init = self.arm.connect(ip)
        if init == 0:
            raise ConnectionError("Failed to connect to MARVIN robot")
        
        time.sleep(0.5)
        self.arm.clear_set()
        self.arm.clear_error('A')
        self.arm.clear_error('B')
        self.arm.send_cmd()
        time.sleep(0.5)
        
        # 验证连接
        motion_tag = 0
        frame_update = None
        for i in range(10):
            sub_data = self.arm.subscribe(self.dcss)
            frame_serial = sub_data['outputs'][0]['frame_serial']
            if frame_serial != 0 and frame_update != frame_serial:
                motion_tag += 1
                frame_update = frame_serial
            time.sleep(0.1)
        
        if motion_tag == 0:
            raise ConnectionError("Robot connection verification failed")
        
        # 加载运动学配置
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        ini_result = self.kine.load_config(arm_type=ARM_TYPE, config_path=config_file)
        if not ini_result:
            raise RuntimeError("Failed to load kinematics config")
        
        self.kine.initial_kine(
            robot_type=ini_result['TYPE'][0],
            dh=ini_result['DH'][0],
            pnva=ini_result['PNVA'][0],
            j67=ini_result['BD'][0]
        )
        
        # 设置控制模式
        self.arm.clear_set()
        self.arm.set_state(arm=ARM_NAME, state=1)  # 位置跟随模式
        self.arm.set_vel_acc(arm=ARM_NAME, velRatio=VEL_RATIO_FAST, AccRatio=ACC_RATIO)
        self.arm.send_cmd()
        time.sleep(0.5)
        
        # 清空485缓存并激活夹爪
        self.arm.clear_485_cache(ARM_NAME)
        time.sleep(0.5)
        self._activate_gripper()
        
        # 相机初始化
        self.caps = {}
        for name, idx in camera_indices.items():
            cap = cv2.VideoCapture(idx)
            cap.set(3, 640)
            cap.set(4, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.caps[name] = cap
        
        self.current_gripper_state = 0.0
        time.sleep(1.5)
        print("MARVIN robot connected.")

    def _calc_crc16(self, data: bytearray) -> int:
        """计算Modbus RTU CRC16校验码"""
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for i in range(8):
                if (crc & 1) != 0:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc

    def _send_modbus_cmd(self, payload: list):
        """构建并发送Modbus指令"""
        data_bytes = bytearray(payload)
        crc = self._calc_crc16(data_bytes)
        data_bytes.append(crc & 0xFF)
        data_bytes.append((crc >> 8) & 0xFF)
        hex_str = " ".join([f"{b:02X}" for b in data_bytes])
        hex_clean = hex_str.replace(' ', '')
        byte_len = len(hex_clean) // 2
        
        success, _ = self.arm.set_485_data(
            ARM_NAME, hex_str, byte_len, GRIPPER_COM_PORT
        )
        
        if not success:
            print(f"[Error] 夹爪指令发送失败: {hex_str}")
        
        time.sleep(0.05)

    def _activate_gripper(self):
        """激活/使能夹爪"""
        print("正在激活夹爪...")
        cmd_reset = [
            GRIPPER_SLAVE_ID, 0x10, 0x03, 0xE8, 0x00, 0x01, 0x02, 0x00, 0x00
        ]
        self._send_modbus_cmd(cmd_reset)
        time.sleep(0.5)
        
        cmd_enable = [
            GRIPPER_SLAVE_ID, 0x10, 0x03, 0xE8, 0x00, 0x01, 0x02, 0x00, 0x01
        ]
        self._send_modbus_cmd(cmd_enable)
        print("激活指令已发送，等待夹爪初始化(约2秒)...")
        time.sleep(2.0)
        self.gripper_activated = True

    def _gripper_move(self, position, speed=255, force=255):
        """控制夹爪移动"""
        if not hasattr(self, 'gripper_activated') or not self.gripper_activated:
            print("[Warning] 夹爪未激活，正在激活...")
            self._activate_gripper()
        
        pos = int(position) & 0xFF
        spd = int(speed) & 0xFF
        frc = int(force) & 0xFF
        
        payload = [
            GRIPPER_SLAVE_ID, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
            0x00, 0x09, pos, 0x00, frc, spd
        ]
        self._send_modbus_cmd(payload)

    def close_gripper(self):
        """夹爪关闭"""
        self._gripper_move(position=255, speed=255, force=255)
        self.current_gripper_state = 1.0 
        time.sleep(0.5)

    def open_gripper(self):
        """夹爪打开"""
        self._gripper_move(position=0, speed=255, force=255)
        self.current_gripper_state = 0.0 
        time.sleep(0.5)

    def flush_cameras(self):
        """清空相机缓冲区"""
        for cap in self.caps.values():
            for _ in range(4):
                cap.grab()

    def get_observation(self) -> dict:
        """获取观测数据"""
        obs = {}
        for name, cap in self.caps.items():
            ret, frame = cap.read()
            if not ret: frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if name in CROP_CONFIGS:
                x, y, w, h = CROP_CONFIGS[name]
                frame = frame[y:y+h, x:x+w]
            obs[name] = image_tools.convert_to_uint8(frame)
        
        # 如果缺少 cam_left_wrist，用黑图替代（与 cam_right_wrist 相同尺寸）
        if "cam_left_wrist" not in obs:
            if "cam_right_wrist" in obs:
                # 创建与 cam_right_wrist 相同尺寸的黑图
                right_shape = obs["cam_right_wrist"].shape
                obs["cam_left_wrist"] = np.zeros(right_shape, dtype=obs["cam_right_wrist"].dtype)
            else:
                # 如果没有右腕相机，使用默认尺寸
                obs["cam_left_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 如果缺少 cam_right_wrist，用黑图替代（与 cam_left_wrist 相同尺寸）
        if "cam_right_wrist" not in obs:
            if "cam_left_wrist" in obs:
                left_shape = obs["cam_left_wrist"].shape
                obs["cam_right_wrist"] = np.zeros(left_shape, dtype=obs["cam_left_wrist"].dtype)
            else:
                # 如果两个都没有，使用默认尺寸
                obs["cam_right_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        obs["cam_high"] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 获取关节角度（度转弧度）
        sub_data = self.arm.subscribe(self.dcss)
        arm_idx = 0 if ARM_NAME == 'A' else 1
        joints_deg = sub_data['outputs'][arm_idx]['fb_joint_pos']
        joints_rad = [np.radians(j) for j in joints_deg[:6]]
        obs["state"] = np.append(joints_rad, self.current_gripper_state)
        return obs

    def get_current_cartesian(self):
        """获取当前绝对坐标 [米, 度]"""
        sub_data = self.arm.subscribe(self.dcss)
        arm_idx = 0 if ARM_NAME == 'A' else 1
        current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
        
        curr_pose_mat = self.kine.fk(joints=current_joints)
        if not curr_pose_mat: return None
        
        curr_xyzabc = self.kine.mat4x4_to_xyzabc(pose_mat=curr_pose_mat)
        # [毫米, 度] -> [米, 度]
        pose = np.array(curr_xyzabc, dtype=np.float32)
        pose[:3] /= 1000.0 
        # 注意：这里千万不要再转弧度了！
        return pose

    def move_to_joints(self, joints, vel_ratio=None, wait=True):
        """移动到指定关节角度"""
        if vel_ratio is None:
            vel_ratio = VEL_RATIO_FAST
        
        self.arm.clear_set()
        self.arm.set_state(arm=ARM_NAME, state=1)
        self.arm.set_vel_acc(arm=ARM_NAME, velRatio=vel_ratio, AccRatio=ACC_RATIO)
        self.arm.set_joint_cmd_pose(arm=ARM_NAME, joints=joints)
        self.arm.send_cmd()
        
        if wait:
            time.sleep(2.0)
        
        return True

    def move_to(self, pos, vel_ratio=None, wait=True):
        """移动到指定笛卡尔位姿 [x, y, z, r, p, y] (单位: 毫米, 度)"""
        if vel_ratio is None:
            vel_ratio = VEL_RATIO_FAST
        
        # 获取当前关节角度作为参考
        sub_data = self.arm.subscribe(self.dcss)
        arm_idx = 0 if ARM_NAME == 'A' else 1
        current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
        
        # 计算逆运动学
        target_pose_mat = self.kine.xyzabc_to_mat4x4(xyzabc=pos)
        ik_para = FX_InvKineSolvePara()
        mat16 = self.kine.mat4x4_to_mat1x16(target_pose_mat)
        ik_para.set_input_ik_target_tcp(mat16)
        ik_para.set_input_ik_ref_joint(current_joints)
        ik_para.set_input_ik_zsp_type(0)
        
        ik_result = self.kine.ik(structure_data=ik_para)
        
        if not ik_result or ik_result.m_Output_IsOutRange or ik_result.m_Output_IsJntExd:
            print(f"[Error] IK failed or out of range. Target: {pos}")
            return False
        
        target_joints = ik_result.m_Output_RetJoint.to_list()
        
        # 设置速度和加速度
        self.arm.clear_set()
        self.arm.set_state(arm=ARM_NAME, state=1)
        self.arm.set_vel_acc(arm=ARM_NAME, velRatio=vel_ratio, AccRatio=ACC_RATIO)
        self.arm.set_joint_cmd_pose(arm=ARM_NAME, joints=target_joints)
        self.arm.send_cmd()
        
        if wait:
            time.sleep(2.0)
        
        return True

    def execute_action(self, action_delta):
        """
        执行单步动作 (Cartesian Delta Mode)
        修正版：解决通信协议导致机械臂不动的问题
        """
        # 1. 获取当前绝对位姿 (单位: 米, 度)
        curr_pose = self.get_current_cartesian()
        if curr_pose is None: return False
        
        # 2. 计算目标绝对位姿 (Current + Delta)
        # 注意：这里取 action_delta 的前6位进行计算，忽略夹爪
        target_pose = curr_pose + action_delta[:6]
        
        # 3. Z轴安全限位 (单位: 米)
        target_z_mm = target_pose[2] * 1000.0
        if target_z_mm < MIN_SAFE_Z:
            # print(f"[Safety] Limit Z: {target_z_mm:.1f} -> {MIN_SAFE_Z}")
            target_pose[2] = MIN_SAFE_Z / 1000.0

        # 4. 转换为 SDK 需要的 [毫米, 度]
        target_xyzabc = target_pose.copy()
        target_xyzabc[:3] *= 1000.0  # 米 -> 毫米
        # target_xyzabc[3:] 保持不变 (已经是度了)
        
        # 5. IK 解算
        sub_data = self.arm.subscribe(self.dcss)
        arm_idx = 0 if ARM_NAME == 'A' else 1
        current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
        current_joints = [float(j) for j in current_joints]
        
        target_pose_mat = self.kine.xyzabc_to_mat4x4(xyzabc=target_xyzabc)
        ik_para = FX_InvKineSolvePara()
        mat16 = self.kine.mat4x4_to_mat1x16(target_pose_mat)
        ik_para.set_input_ik_target_tcp(mat16)
        ik_para.set_input_ik_ref_joint(current_joints)
        ik_para.set_input_ik_zsp_type(0)
        
        ik_result = self.kine.ik(structure_data=ik_para)
        
        target_joints = []
        valid_solution_found = False
        
        # 鲁棒的 IK 检查逻辑
        if ik_result:
            temp_joints = ik_result.m_Output_RetJoint.to_list()
            # 只要算出非零解，就认为是有效的
            if not all(abs(j) < 1e-3 for j in temp_joints):
                target_joints = temp_joints
                valid_solution_found = True
        
        if not valid_solution_found:
            print("[Error] IK Failed. Target unreachable.")
            return False
        
        # 6. 检查关节跳变 (阈值设为 45度)
        curr_j = np.array(current_joints[:7])
        targ_j = np.array(target_joints[:7])
        diff = np.max(np.abs(curr_j - targ_j))
        if diff > 45.0:
            print(f"!!! DANGER: Joint jump too large ({diff:.2f}°)! Stop!")
            return False
        
        # 7. 插值执行 (关键修正部分)
        duration = (1.0 / CONTROL_FREQ) * SLOW_DOWN_FACTOR
        steps = int(duration * INTERPOLATION_FREQ)
        if steps < 1: steps = 1
        
        for i in range(1, steps + 1):
            alpha = i / steps
            interp = curr_j + (targ_j - curr_j) * alpha
            
            # === [修复核心] 必须在每一步都重新设置状态和速度 ===
            self.arm.clear_set()
            self.arm.set_state(arm=ARM_NAME, state=1) # 1: 伺服/透传模式
            # 建议稍微调高这里的速度，防止插值跟不上导致卡顿 (10 -> 30)
            self.arm.set_vel_acc(arm=ARM_NAME, velRatio=30, AccRatio=10) 
            self.arm.set_joint_cmd_pose(arm=ARM_NAME, joints=interp.tolist())
            self.arm.send_cmd()
            # ===============================================
            
            time.sleep(1.0 / INTERPOLATION_FREQ)

        # 8. 夹爪 (取第7维)
        target_gripper = action_delta[6]
        if target_gripper > 0.8: self.close_gripper()
        elif target_gripper < 0.2: self.open_gripper()
        
        return True

    def find_reachable_ik(self, start_pose, end_pose, search_steps=5):
        """如果在 end_pose IK 失败，则在 start 和 end 之间二分查找最近的可达点"""
        def get_ik(p):
            ik_p = p.copy()
            ik_p[:3] *= 1000.0
            ik_p[3:] = np.degrees(ik_p[3:])
            target_pose_mat = self.kine.xyzabc_to_mat4x4(xyzabc=ik_p)
            ik_para = FX_InvKineSolvePara()
            mat16 = self.kine.mat4x4_to_mat1x16(target_pose_mat)
            ik_para.set_input_ik_target_tcp(mat16)
            sub_data = self.arm.subscribe(self.dcss)
            arm_idx = 0 if ARM_NAME == 'A' else 1
            current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
            ik_para.set_input_ik_ref_joint(current_joints)
            ik_para.set_input_ik_zsp_type(0)
            ik_result = self.kine.ik(structure_data=ik_para)
            if ik_result and not ik_result.m_Output_IsOutRange and not ik_result.m_Output_IsJntExd:
                return True, ik_result.m_Output_RetJoint.to_list()
            return False, None

        ret, joints = get_ik(end_pose)
        if ret:
            return ret, joints, end_pose

        print(f"[Warning] Original IK Failed. Searching for nearest reachable point...")
        for ratio in [0.75, 0.5, 0.25, 0.1]:
            temp_pose = start_pose + (end_pose - start_pose) * ratio
            ret, joints = get_ik(temp_pose)
            if ret:
                print(f"[Recovery] Found reachable point at {ratio*100:.0f}% of original step.")
                return ret, joints, temp_pose

        return -1, None, None

    def move_home_scripted(self):
        """移动到home点（使用关节角度控制）"""
        self.move_to_joints(HOME_JOINTS, vel_ratio=VEL_RATIO_FAST, wait=True)

    def move_to_start(self, target_action_rad):
        """从全0初始位姿移动到home点（使用关节角度控制）"""
        # 先移动到全0
        self.move_to_joints(INIT_JOINTS, vel_ratio=VEL_RATIO_FAST, wait=True)
        # 再移动到home点
        self.move_to_joints(HOME_JOINTS, vel_ratio=VEL_RATIO_FAST, wait=True)
        time.sleep(0.5)

    def run_setup(self, target_pose):
        """设置流程：移动到目标点放置物体"""
        # 检查 POS_A 是否已设置
        if POS_A is None:
            print("[Error] POS_A 未初始化！无法执行 setup。")
            self.move_home_scripted()
            return
        
        # 确保 POS_A 是列表格式且单位正确（毫米，度）
        pose_A = list(POS_A)
        if len(pose_A) != 6:
            print(f"[Error] POS_A 格式错误: {pose_A}")
            self.move_home_scripted()
            return
        
        # 检查单位：如果 Z 坐标小于 1，可能是米单位，需要转换
        if abs(pose_A[2]) < 1.0:
            print(f"[Warning] POS_A Z坐标 {pose_A[2]} 看起来像是米单位，转换为毫米...")
            pose_A[:3] = [p * 1000.0 for p in pose_A[:3]]
            pose_A[3:] = [np.degrees(p) for p in pose_A[3:]]  # 弧度转度
        
        pose_A_up = list(pose_A)
        pose_A_up[2] += 100  # Z轴向上100mm
        target_up = list(target_pose)
        target_up[2] += 100
        
        print(f"[Setup] POS_A: {pose_A}")
        print(f"[Setup] pose_A_up: {pose_A_up}")
        print(f"[Setup] target_pose: {target_pose}")
        print(f"[Setup] target_up: {target_up}")
        
        # try:
        #     self.move_to(pose_A_up, vel_ratio=VEL_RATIO_FAST, wait=True)
        #     self.move_to(pose_A, vel_ratio=VEL_RATIO_FAST, wait=True)
        #     self.open_gripper()
        #     time.sleep(2.0)
        #     self.close_gripper()
        #     time.sleep(2.0)
        #     self.move_to(pose_A_up, vel_ratio=VEL_RATIO_FAST, wait=True)
        #     self.move_home_scripted()
        #     self.move_to(target_up, vel_ratio=VEL_RATIO_FAST, wait=True)
        #     self.move_to(target_pose, vel_ratio=VEL_RATIO_FAST, wait=True)
        #     self.open_gripper()
        #     time.sleep(1.5)
        #     self.move_to(target_up, vel_ratio=VEL_RATIO_FAST, wait=True)
        #     self.move_home_scripted()
        # except Exception as e:
        #     print(f"[Setup Error] {e}")
        #     self.move_home_scripted()

    def joints_to_cartesian(self, joints_deg):
        """
        将关节角度转换为笛卡尔坐标
        :param joints_deg: 关节角度列表（度），前6个关节
        :return: 笛卡尔坐标 [x, y, z, r, p, y] (单位: 毫米, 度)
        """
        # joints = joints_deg[:6]  # 只取前6个关节
        pose_mat = self.kine.fk(joints=joints_deg)
        if not pose_mat:
            return None
        
        xyzabc = self.kine.mat4x4_to_xyzabc(pose_mat=pose_mat)
        return np.array(xyzabc, dtype=np.float32)  # [x, y, z, r, p, y] 单位：毫米，度
    
    def is_in_boundary(self, pose_mm, boundary_points):
        """检查笛卡尔坐标(mm)是否在2D凸包范围内"""
        pt = (float(pose_mm[0]), float(pose_mm[1]))
        contour = boundary_points.astype(np.float32)
        result = cv2.pointPolygonTest(contour, pt, False)
        return result >= 0

    def recover_from_error(self, target_mode=1):
        """恢复错误状态"""
        print(f"\n[Recovery] !!! 启动自动恢复程序")
        if self.arm is None: return
        
        self.arm.clear_set()
        self.arm.clear_error('A')
        self.arm.clear_error('B')
        self.arm.send_cmd()
        time.sleep(0.5)
        
        self.arm.clear_set()
        self.arm.set_state(arm=ARM_NAME, state=target_mode)
        self.arm.set_vel_acc(arm=ARM_NAME, velRatio=VEL_RATIO_SERVO, AccRatio=ACC_RATIO)
        self.arm.send_cmd()
        time.sleep(0.5)
        print(f"[Recovery] 恢复完成")

    def close(self):
        """关闭连接"""
        if hasattr(self, 'arm'):
            self.arm.set_state(arm=ARM_NAME, state=0)
            self.arm.send_cmd()
            self.arm.release_robot()
        for cap in self.caps.values():
            cap.release()


# -----------------------------------------------------------------------------
# 4. 主程序
# -----------------------------------------------------------------------------
def main():
    print(f"Loading Model: {CONFIG_NAME}...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    robot = MarvinHardware(ROBOT_IP, CAMERAS, CONFIG_FILE)
    
    # 使用正运动学从 HOME_JOINTS 计算 HOME_POS
    global HOME_POS, POS_A, FIXED_Z, FIXED_ROLL, FIXED_PITCH, BASE_YAW
    HOME_POS_calc = robot.joints_to_cartesian(HOME_JOINTS)
    if HOME_POS_calc is not None:
        HOME_POS = HOME_POS_calc.tolist()
        print(f"[Info] 从 HOME_JOINTS 计算出 HOME_POS: {HOME_POS}")
        print(f"       X={HOME_POS[0]:.2f}mm, Y={HOME_POS[1]:.2f}mm, Z={HOME_POS[2]:.2f}mm")
        print(f"       R={HOME_POS[3]:.2f}°, P={HOME_POS[4]:.2f}°, Y={HOME_POS[5]:.2f}°")
        
        # 将 POS_A 设置为与 HOME_POS 相同的值
        POS_A = list(HOME_POS)
        # FIXED_Z = POS_A[2]
        # FIXED_ROLL = POS_A[3]
        # FIXED_PITCH = POS_A[4]
        # BASE_YAW = POS_A[5]
        print(f"[Info] POS_A 已设置为与 HOME_POS 相同: {POS_A}")
    else:
        print("[Warning] 无法计算 HOME_POS，使用默认值")
        HOME_POS = [-169.82, -463.97, 249.69, 177.63, 3.34, -37.77]
        POS_A = list(HOME_POS)
        # FIXED_Z = POS_A[2]
        # FIXED_ROLL = POS_A[3]
        # FIXED_PITCH = POS_A[4]
        # BASE_YAW = POS_A[5]
    
    sampler = TaskSampler(POINTS_FILE, progress_file=PROGRESS_FILE, resume=RESUME_TESTING)
    viz = TaskVisualizer(VIS_SAVE_DIR, RESULT_IMG_NAME, BOUNDARY_POINTS_2D, HOME_POS)
    debugger = DebugVisualizer(MIN_SAFE_Z, VIS_SAVE_DIR)
    recorder = MetricsRecorder()

    kb = KeyboardThread()
    kb.start()
    pause_requested = False
    
    current_target = None
    
    try:
        episode = 0
        
        while True:
            episode += 1
            print(f"\n=== Episode {episode} ===")
            
            # 1. 获取下一个目标点
            target_pose, idx, total = sampler.get_next_target()
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
            start_pose_abs = robot.get_current_cartesian()
            recorder.start_episode(start_pose_abs, target_pose)
            
            # 4. AI 控制循环
            print(">>> AI Loop running... Press 'o' to ABORT (Mark as Fail).")
            aborted = False
            current_timeout_limit = 120.0 if episode == 1 else 27.0
            episode_start_time = time.time()
            consecutive_re_inference_count = 0
            MAX_RETRY = 10
            close_gripper_time = time.time()
            
            while True:
                if kb.get_and_clear_key() == 'o':
                    aborted = True
                    break
                elif kb.get_and_clear_key() == 'p':
                    if not pause_requested:
                        print("\n>>> ⏳ [指令收到] 本轮结束后将暂停...")
                        pause_requested = True
                elif kb.get_and_clear_key() == 'y':
                    print("\n>>> 🎯 [指令收到] 手动触发抓取 (Mark as Success).")
                    robot.close_gripper()
                    time.sleep(0.5)
                    close_gripper_time = time.time()
                    break
                
                # 1. 观测
                raw_obs = robot.get_observation()
                curr_pose_abs = robot.get_current_cartesian()
                
                # 构造相对输入 State
                def normalize_angle(angle):
                    return (angle + np.pi) % (2 * np.pi) - np.pi
                
                rel_pose = curr_pose_abs - start_pose_abs
                rel_pose[5] = normalize_angle(curr_pose_abs[5] - start_pose_abs[5])
                print(f"\r[State] Rel_pose: {rel_pose[:3]}", end="")
                
                input_state = np.append(rel_pose, robot.current_gripper_state)
                
                # 2. 推理
                result = policy.infer({
                    "cam_left_wrist": raw_obs["cam_left_wrist"],
                    "cam_right_wrist": raw_obs["cam_right_wrist"],
                    "state": input_state, "prompt": TASK_PROMOT
                })
                
                action_chunk = np.array(result["actions"])
                debugger.update(raw_obs, action_chunk, robot)
                
                # 3. 抓取检测
                if np.any(action_chunk[:1, 6] > 0.8):
                    close_gripper_time = time.time()
                    print(">>> Auto Grasp Detected.")
                    break
                
                # 4. 执行
                steps_to_run = min(EXECUTE_STEPS, len(action_chunk))
                need_re_inference = False
                key = None
                for i in range(steps_to_run):
                    current_duration = time.time() - episode_start_time
                    if current_duration > current_timeout_limit:
                        print(f"\n[Timeout] 耗时 {current_duration:.1f}s > {current_timeout_limit:.1f}s. 强制中断...")
                        aborted = True
                        break

                    raw_action = action_chunk[i]
                    pred_target_abs = start_pose_abs[:6] + raw_action[:6]
                    pred_target_mm = pred_target_abs * 1000.0
                    
                    # 边界检查
                    if not robot.is_in_boundary(pred_target_mm, BOUNDARY_EXPANDED):
                        print(f"\n[Safety] 目标 ({pred_target_mm[0]:.0f}, {pred_target_mm[1]:.0f}) 超出宽松边界！正在回拉...")
                        center_pt = np.mean(BOUNDARY_POINTS_2D, axis=0)
                        curr_xy = pred_target_mm[:2]
                        vec_to_center = center_pt - curr_xy
                        norm = np.linalg.norm(vec_to_center)
                        
                        if norm > 0:
                            pull_back_vec = (vec_to_center / norm) * 150.0
                        else:
                            pull_back_vec = np.array([10.0, 10.0])
                        
                        curr_pose_recover = robot.get_current_cartesian()
                        target_pose_recover = curr_pose_recover.copy()
                        target_pose_recover[0] += pull_back_vec[0] / 1000.0
                        target_pose_recover[1] += pull_back_vec[1] / 1000.0
                        
                        delta_recover = target_pose_recover - curr_pose_recover
                        action_recover = np.append(delta_recover[:6], robot.current_gripper_state)
                        
                        print(f"[Recovery] 执行回拉动作: dX={pull_back_vec[0]:.1f}mm, dY={pull_back_vec[1]:.1f}mm")
                        robot.execute_action(action_recover)
                        need_re_inference = True
                        break
                    
                    # 当前绝对位置
                    curr_pose_abs = robot.get_current_cartesian()
                    real_delta = pred_target_abs - curr_pose_abs
                    action_to_execute = np.append(real_delta, raw_action[6])
                    
                    success = robot.execute_action(action_to_execute)
                    if not success:
                        print("\n[Safety] 动作执行失败 (关节跳变/IK)。启动【主动恢复】策略...")
                        curr_pose_m = robot.get_current_cartesian()
                        curr_xy_mm = curr_pose_m[:2] * 1000.0
                        center_pt = np.mean(BOUNDARY_POINTS_2D, axis=0)
                        vec_to_center = center_pt - curr_xy_mm
                        dist_to_center = np.linalg.norm(vec_to_center)
                        
                        recovery_delta = np.zeros(7)
                        if dist_to_center > 0:
                            direction = vec_to_center / dist_to_center
                            recovery_delta[0] = direction[0] * 0.03
                            recovery_delta[1] = direction[1] * 0.03
                        recovery_delta[2] = 0.02
                        recovery_delta[6] = robot.current_gripper_state
                        
                        print(f"[Recovery] 执行避险动作: 向中心回拉 3cm, 向上抬起 2cm...")
                        rec_success = robot.execute_action(recovery_delta)
                        
                        if rec_success:
                            print("[Recovery] 避险动作执行成功。重新开始推理。")
                        else:
                            print("[Recovery] 避险动作也失败了 (可能卡死)。")
                        
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
                        break
                    
                    curr_pos_abs = robot.get_current_cartesian()
                    recorder.step(curr_pos_abs)
                
                if aborted: break
                if key == 'y':
                    break
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
                viz.update_result(target_pose, False)
                robot.open_gripper()
                target_up = list(target_pose)
                target_up[2] += 100
                robot.move_to(target_up, vel_ratio=VEL_RATIO_FAST, wait=True)
                target_pose_withObjectZ = list(target_pose)
                target_pose_withObjectZ[2] = OBJECT_Z
                robot.move_to(target_pose_withObjectZ, vel_ratio=VEL_RATIO_FAST, wait=True)
                robot.close_gripper()
                time.sleep(2.0)
                robot.move_to(target_up, vel_ratio=VEL_RATIO_FAST, wait=True)
                current_metrics = recorder.end_episode(success=False, close_gripper_time=close_gripper_time)
                robot.move_to(HOME_POS, vel_ratio=VEL_RATIO_FAST, wait=True)
            else:
                robot.close_gripper()
                time.sleep(2.0)
                current_metrics = recorder.end_episode(success=True, close_gripper_time=close_gripper_time)
                robot.move_home_scripted()
                print(">>> Marked as SUCCESS.")
                viz.update_result(target_pose, True)
            
            sampler.mark_current_done()
            recorder.print_current_metrics(current_metrics)
            recorder.print_summary()
            
            if pause_requested:
                print("\n" + "="*50)
                print(">>> ⏸️  程序已暂停 (用户请求)。")
                print(">>> ⌨️  请按 [ENTER] 键继续下一轮测试...")
                print("="*50)
                
                while True:
                    k = kb.get_and_clear_key()
                    if k == '\n' or k == '\r':
                        print(">>> ▶️  继续运行...")
                        pause_requested = False
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
