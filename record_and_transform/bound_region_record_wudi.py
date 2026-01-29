import argparse
import time
import json
import threading
import sys
import random
import numpy as np
import cv2
import queue
from pathlib import Path
from scipy.spatial import ConvexHull  # 用于构建凸包

# ---------------------------------------------------------
# 硬件导入
# ---------------------------------------------------------
try:
    from xarm.wrapper import XArmAPI
except ImportError:
    print("Error: xarm-python-sdk not found.")
    XArmAPI = None

class AutoDataRecorder:
    def __init__(self, ip, output_dir, camera_indices, crop_configs=None):
        self.ip = ip
        self.output_dir = Path(output_dir)
        self.camera_indices = camera_indices
        self.arm = None
        
        # --- 相机与图像缓存 ---
        self.caps = {}
        self.crop_configs = crop_configs if crop_configs else {}
        self.latest_frames = {}           # 存储每个相机最新一帧的字典
        self.frame_lock = threading.Lock() # 读写锁
        self.camera_running = True        # 控制后台相机线程的标志位
        
        # --- 录制状态 ---
        self.is_recording = False
        self.stop_event = threading.Event()
        self.current_gripper_state = 0.0 
        self.pause_requested = False
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- 关键位姿定义 (单位: 毫米/弧度) ---
        self.pos_home = [486.626923, 158.343277, 30.431152, 3.12897, 0.012689, -1.01436]
        self.pos_A = [486.626923, 158.343277, -69.431152, 3.12897, 0.012689, -1.01436]
        # self.instruction = "pick up the industrial components"
        self.instruction =  "pick up the hollow rectangular housing"
        self.fixed_z = self.pos_A[2]  # Z轴固定为桌面高度

        # ===================== 2D凸包区域定义 =====================
        self.boundary_points_2d = np.array([
            [528.6, 126.5],
            [745.0, 250.2],
            [501.9, 539.4],
            [338.1, 425.0],
        ])
        scale_factor = 0.8  # 缩放比例，0.8 表示缩小到boundary_points_2d的 80%，根据需求调整
        center_point = np.mean(self.boundary_points_2d, axis=0)
        # 将每个顶点向中心点靠拢
        self.boundary_points_2d = center_point + (self.boundary_points_2d - center_point) * scale_factor
        
        # 1. 构建凸包
        self.hull_2d = ConvexHull(self.boundary_points_2d)
        
        # 2. 获取逆时针顶点 (ConvexHull 默认是 CCW)
        ccw_indices = self.hull_2d.vertices
        self.ccw_vertices = self.boundary_points_2d[ccw_indices]
        
        # 3. 预计算路径 (生成沿着边界的密集点)
        self.boundary_step_size = 100.0  # 步长 20mm
        self.path_points_2d = self._generate_boundary_path(self.ccw_vertices, self.boundary_step_size)
        
        # 4. 路径控制变量
        self.current_path_idx = 0
        self.total_path_points = len(self.path_points_2d)
        
        # 5. 方向控制变量 (1: CCW, -1: CW)
        self.path_direction = 1 

        print(f">>> [Sampler] Boundary Path Generated. Total Points: {self.total_path_points}")
        
        # 计算包围盒（用于快速采样）
        self.min_x = np.min(self.boundary_points_2d[:, 0])
        self.max_x = np.max(self.boundary_points_2d[:, 0])
        self.min_y = np.min(self.boundary_points_2d[:, 1])
        self.max_y = np.max(self.boundary_points_2d[:, 1])
        
        # ===================== 姿态随机化 =====================
        self.fixed_roll = self.pos_A[3]
        self.fixed_pitch = self.pos_A[4]
        
        # Yaw角度随机范围 (弧度)
        self.base_yaw = self.pos_A[5]
        self.yaw_random_range = (-np.pi/2, np.pi/6)  # ±45度
        self.current_yaw_angle = self.base_yaw

        # 运动速度
        self.speed_fast = 500 
        self.speed_record = 100 
        self.speed_adjust = 50 
        self.joint_speed_fast = 0.5 
        self.joint_speed_record = 0.15
        
        # === 数据队列 ===
        self.data_queue = queue.Queue(maxsize=50)
        
        # === 扰动参数 ===
        self.perturbation_prob = 0.2  # 60% 的概率加入扰动（故意走偏再修正）
        self.perturbation_range = 30.0 # 扰动范围 30mm (3cm)

    # ---------------------------------------------------------
    # 【新增】生成边界路径
    # ---------------------------------------------------------
    def _generate_boundary_path(self, vertices, step_size):
        """生成沿着多边形边界的密集点序列"""
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

    # ---------------------------------------------------------
    # 【修改】获取下一个边界点
    # ---------------------------------------------------------
    def get_next_boundary_target(self):
        """
        获取下一个边界点作为目标。
        根据 self.path_direction 决定是下一个还是上一个。
        """
        max_attempts = self.total_path_points + 10
        attempts = 0
        
        while attempts < max_attempts:
            # 1. 取出当前索引的点
            target_2d = self.path_points_2d[self.current_path_idx]
            
            target_x = target_2d[0]
            target_y = target_2d[1]
            target_z = self.fixed_z
            
            # 2. 尝试随机 Yaw
            for _ in range(10):
                yaw_noise = random.uniform(*self.yaw_random_range)
                candidate_yaw = self.base_yaw + yaw_noise
                
                candidate_pose = [
                    target_x, target_y, target_z, 
                    self.fixed_roll, self.fixed_pitch, candidate_yaw
                ]

                # 3. 可达性检查
                if self.check_pose_reachable(candidate_pose):
                    dir_str = "CCW" if self.path_direction == 1 else "CW"
                    print(f"  [Target] {dir_str} Point {self.current_path_idx}/{self.total_path_points} -> OK")
                    self.current_yaw_angle = candidate_yaw
                    
                    # 准备下一次调用的索引 (循环移动)
                    self.current_path_idx = (self.current_path_idx + self.path_direction) % self.total_path_points
                    
                    return candidate_pose
            
            # 如果不可达，尝试下一个点
            print(f"  [Warn] Point {self.current_path_idx} Unreachable. Skipping...")
            self.current_path_idx = (self.current_path_idx + self.path_direction) % self.total_path_points
            attempts += 1
            
        print("[Error] 边界路径均不可达，返回中心保底。")
        region_center_2d = np.mean(self.boundary_points_2d, axis=0)
        return [region_center_2d[0], region_center_2d[1], self.fixed_z, 
                self.fixed_roll, self.fixed_pitch, self.base_yaw]

    # ... (get_random_start_pose, is_point_inside_hull_2d 等保持不变) ...
    def get_random_start_pose(self):
        """生成一个随机起始点 (Home)"""
        center = np.mean(self.boundary_points_2d, axis=0)
        home_shrink_ratio = 0.8 
        for _ in range(50):
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            point = np.array([x, y])
            test_point_for_check = center + (point - center) / home_shrink_ratio
            if self.is_point_inside_hull_2d(test_point_for_check, self.hull_2d):
                z = self.pos_home[2] + random.uniform(-5.0, 20.0)
                yaw = self.base_yaw + random.uniform(-0.1, 0.1)
                return [x, y, z, self.fixed_roll, self.fixed_pitch, yaw]
        return self.pos_home

    def is_point_inside_hull_2d(self, point_2d, hull_2d):
        hull_eq = hull_2d.equations
        point_homo = np.hstack([point_2d, 1])
        for eq in hull_eq:
            if np.dot(eq, point_homo) > 1e-6:
                return False
        return True

    # ---------------------------------------------------------
    # 机器人控制 (保持不变)
    # ---------------------------------------------------------
    def connect_robot(self):
        if XArmAPI is None: raise ImportError("xArm SDK Missing")
        print(f"Connecting to xArm at {self.ip}...")
        self.arm = XArmAPI(self.ip)
        self.arm.clean_error()
        self.arm.clean_warn()
        time.sleep(1.0) 
        self.arm.motion_enable(enable=True)
        time.sleep(1.0) 
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        time.sleep(1.0) 
        self.arm.set_tgpio_modbus_baudrate(baud=115200)
        time.sleep(1.0) 
        print("xArm connected.")

    def close_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 16, 1, 2, 0, 2, 4, 0, 0, 46, 224])
        self.arm.getset_tgpio_modbus_data([0x01, 6, 1, 8, 0, 1])
        self.current_gripper_state = 1.0 

    def open_gripper(self):
        self.arm.getset_tgpio_modbus_data([0x01, 16, 1, 2, 0, 2, 4, 0, 0, 0, 0])
        self.arm.getset_tgpio_modbus_data([0x01, 6, 1, 8, 0, 1])
        self.current_gripper_state = 0.0 

    def clear_robot_error(self):
        print("!!! 检测到机械臂错误，正在尝试自动恢复...")
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        time.sleep(1.0)
        self.move_to(self.pos_home, speed=self.joint_speed_fast)
        
    def move_to(self, pos, speed=None, wait=True):
        if speed is None: speed = self.speed_fast
        ret = self.arm.set_position(x=pos[0], y=pos[1], z=pos[2], 
                                    roll=pos[3], pitch=pos[4], yaw=pos[5], 
                                    speed=speed, wait=wait, is_radian=True)
        if ret != 0:
            print(f"[Error] set_position failed. Target: {pos}")
            self.clear_robot_error()
            return False
        return True
    
    def move_to_joint_by_pose(self, target_pose, speed=None, wait=True):
        code, joint_angles = self.arm.get_inverse_kinematics(target_pose, input_is_radian=True, return_is_radian=True)
        if code != 0: return False
        ret = self.arm.set_servo_angle(angle=joint_angles, speed=speed, wait=wait, is_radian=True)
        if ret != 0:
            self.clear_robot_error()
            return False
        return True

    def check_pose_reachable(self, pose):
        code, _ = self.arm.get_inverse_kinematics(pose, input_is_radian=True, return_is_radian=True)
        return code == 0

    # ... (相机和写入线程部分保持不变) ...
    def connect_cameras(self):
        for name, idx in self.camera_indices.items():
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.caps[name] = cap
            else:
                print(f"[Error] 无法打开相机 {name} (Index {idx})")
        print(f"Connected to {len(self.caps)} cameras.")
        self.cam_thread = threading.Thread(target=self._camera_daemon, daemon=True)
        self.cam_thread.start()
        print(">>> 正在预热相机 (2秒)...")
        time.sleep(2.0)

    def _camera_daemon(self):
        while self.camera_running:
            for name, cap in self.caps.items():
                ret, frame = cap.read()
                if ret:
                    with self.frame_lock:
                        self.latest_frames[name] = frame
            time.sleep(0.002) 

    def _get_latest_images(self):
        images = {}
        with self.frame_lock:
            raw_frames = self.latest_frames.copy()
        for name, frame in raw_frames.items():
            if name in self.crop_configs and self.crop_configs[name] is not None:
                x, y, w, h = self.crop_configs[name]
                frame = frame[y:y+h, x:x+w]
            images[name] = frame
        return images

    def _writer_worker(self, episode_dir):
        state_file = episode_dir / "data.jsonl"
        with open(state_file, "a") as f:
            while True:
                try:
                    item = self.data_queue.get(timeout=2)
                    if item is None:
                        self.data_queue.task_done()
                        break
                    data_dict, images_dict = item
                    f.write(json.dumps(data_dict) + "\n")
                    frame_idx = data_dict["frame_idx"]
                    for cam_name, img_data in images_dict.items():
                        save_path = episode_dir / "images" / cam_name / f"{frame_idx:06d}.jpg"
                        cv2.imwrite(str(save_path), img_data)
                    self.data_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Writer Error] {e}")

    def _recording_thread(self, episode_dir, instruction):
        frame_idx = 0
        target_interval = 0.1  # 5Hz
        print(f">>> [REC] 启动录制: {episode_dir.name}")
        writer = threading.Thread(target=self._writer_worker, args=(episode_dir,))
        writer.start()
        code, last_joint_pos = self.arm.get_servo_angle(is_radian=True)
        if code != 0: last_joint_pos = [0.0]*7

        while not self.stop_event.is_set():
            loop_start = time.time()
            try:
                current_images = self._get_latest_images()
                if not current_images and frame_idx == 0:
                    time.sleep(0.01)
                    continue

                code, curr_joint_pos = self.arm.get_servo_angle(is_radian=True)
                ret, curr_cart_pos = self.arm.get_position(is_radian=True)
                
                if ret != 0 or code != 0:
                    time.sleep(0.005); continue
                
                delta_joints = np.array(curr_joint_pos) - np.array(last_joint_pos)
                delta_joints = np.round(delta_joints, 6).tolist()
                
                data_dict = {
                    "timestamp": time.time(),
                    "frame_idx": frame_idx,
                    "instruction": instruction,
                    "joint_pos": delta_joints,
                    "joint_abs": curr_joint_pos,
                    "cartesian_pos": curr_cart_pos,
                    "gripper_state": self.current_gripper_state
                }
                
                self.data_queue.put((data_dict, current_images))
                last_joint_pos = curr_joint_pos
                
                if frame_idx % 10 == 0:
                    q_size = self.data_queue.qsize()
                    sys.stdout.write(f"\r[REC] Frame: {frame_idx} | Q_Size: {q_size}")
                    sys.stdout.flush()
                frame_idx += 1
                
                elapsed = time.time() - loop_start
                sleep_time = max(0, target_interval - elapsed)
                time.sleep(sleep_time)
            except Exception as e:
                print(f"Record error: {e}")
                break
        
        self.data_queue.put(None)
        writer.join()
        print(f"\n>>> [REC] 录制结束。总帧数: {frame_idx}")

    def start_recording(self, episode_idx, instruction):
        episode_dir = self.output_dir / f"episode_{episode_idx}"
        episode_dir.mkdir(exist_ok=True)
        for cam_name in self.caps.keys():
            (episode_dir / "images" / cam_name).mkdir(parents=True, exist_ok=True)
        self.stop_event.clear()
        while not self.data_queue.empty(): self.data_queue.get()
        self.rec_thread = threading.Thread(target=self._recording_thread, args=(episode_dir, instruction))
        self.rec_thread.start()

    def stop_recording(self):
        if hasattr(self, 'rec_thread') and self.rec_thread.is_alive():
            self.stop_event.set()
            self.rec_thread.join()

    # ---------------------------------------------------------
    # 【修改】输入监听：增加 'a' 键切换方向
    # ---------------------------------------------------------
    def input_listener(self):
        while True:
            try:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'c':
                    self.change_target_requested = True
                    print("\n>>> [指令收到] 本轮结束后将切换到下一个目标点...")
                elif cmd == 'p':
                    self.pause_requested = True
                    print("\n>>> [指令收到] 本轮结束后将暂停...")
                elif cmd == 'a': # 新增：切换方向
                    self.path_direction *= -1
                    dir_str = "顺时针(CW)" if self.path_direction == -1 else "逆时针(CCW)"
                    print(f"\n>>> [指令收到] 方向已切换为: {dir_str}")
            except:
                pass

    # ---------------------------------------------------------
    # 【修改】主流程
    # ---------------------------------------------------------
    def run_auto_collection(self):
        self.connect_robot()
        self.connect_cameras()
        self.change_target_requested = False
        self.pause_requested = False
        threading.Thread(target=self.input_listener, daemon=True).start()

        print("\n" + "="*60)
        print("自动数据采集程序 (Boundary Tracing)")
        print("操作: [c]下一个点 | [a]切换方向 | [p]暂停")
        print("="*60 + "\n")

        # 1. 生成第一个目标点
        current_target_pose = self.get_next_boundary_target()
        safe_target_up = list(current_target_pose); safe_target_up[2] += 100
        
        # 2. 去 pos_A 抓取
        print(">>> [Init] Initial Pickup...")
        self.move_to(self.pos_home, speed=self.speed_fast)
        self.move_to(self.pos_A, speed=self.speed_fast)
        self.close_gripper()
        time.sleep(2.0)
        self.move_to(self.pos_home, speed=self.speed_fast)
        
        # 3. 放到第一个目标点
        self.move_to(safe_target_up, speed=self.speed_fast)
        self.move_to(current_target_pose, speed=self.speed_fast)
        self.open_gripper()
        time.sleep(2.0)
        self.move_to(safe_target_up, speed=self.speed_fast)
        
        print(f">>> [Init] 物体已就位。开始采集。")

        # 自动检测 Index
        existing_episodes = [d.name for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
        if existing_episodes:
            indices = [int(n.split("_")[-1]) for n in existing_episodes if n.startswith("episode_")]
            episode_idx = max(indices) + 1 if indices else 0
        else:
            episode_idx = 0
            
        print(f">>> 从 Episode {episode_idx} 开始")
        
        try:
            while True:
                # --- 暂停 ---
                if self.pause_requested:
                    print("\n" + "!"*40 + "\n>>> 程序已暂停，按回车继续...\n" + "!"*40)
                    input()
                    self.pause_requested = False

                print(f"\n--- Loop {episode_idx} (Idx: {self.current_path_idx}) ---")
                
                # --- 切换目标点 ---
                if self.change_target_requested:
                    print("\n" + "="*40)
                    print(">>> [Change] 切换到下一个边界点...")
                    
                    # 1. 抓起当前物体
                    safe_curr_up = list(current_target_pose); safe_curr_up[2] += 100
                    self.move_to(safe_curr_up, speed=self.speed_fast)
                    self.move_to(current_target_pose, speed=self.speed_fast)
                    self.close_gripper()
                    time.sleep(2.0)
                    self.move_to(safe_curr_up, speed=self.speed_fast)
                    
                    # 2. 生成新目标 (顺/逆时针下一个)
                    new_target_pose = self.get_next_boundary_target()
                    safe_new_up = list(new_target_pose); safe_new_up[2] += 100
                    
                    # 3. 移动并放置
                    self.move_to(safe_new_up, speed=self.speed_fast)
                    self.move_to(new_target_pose, speed=self.speed_fast)
                    self.open_gripper()
                    time.sleep(2.0)
                    self.move_to(safe_new_up, speed=self.speed_fast)
                    
                    current_target_pose = new_target_pose
                    self.change_target_requested = False
                    print(">>> [Change] 切换完成。")
                    print("="*40 + "\n")
                
                # --- 采集流程 ---
                # 1. 随机 Home
                current_home_pos = self.get_random_start_pose()
                print(f"[Home] 随机起始点: ({current_home_pos[0]:.1f}, {current_home_pos[1]:.1f}, {current_home_pos[2]:.1f})")
                
                # 2. 移动到随机 Home (不录制)
                self.move_to(safe_target_up, speed=self.speed_fast) 
                self.move_to_joint_by_pose(current_home_pos, speed=self.speed_fast)
                time.sleep(0.5)
                
                target_pos = current_target_pose
                safe_pos_up = list(target_pos); safe_pos_up[2] += 100
                safe_pos_up[5] = self.current_yaw_angle
                
                # 3. 混合策略采集
                if random.random() > self.perturbation_prob:
                    # Mode A: 标准
                    print("[Prepare] 模式: 标准全流程")
                    time.sleep(1.0)
                    print(f"[Record] 开始录制 Episode {episode_idx} (Full)")
                    self.start_recording(episode_idx, self.instruction)
                    self.move_to(safe_pos_up, speed=self.speed_record)
                    self.move_to(target_pos, speed=self.speed_record)
                
                else:
                    # Mode B: 纠偏
                    perturb_pos = list(target_pos)
                    mode_info = "Correction"

                    if random.random() < 0.7: 
                        target_yaw = target_pos[5]
                        yaw_vec_x = np.cos(target_yaw); yaw_vec_y = np.sin(target_yaw)
                        app_x = target_pos[0] - current_home_pos[0]
                        app_y = target_pos[1] - current_home_pos[1]
                        dot = yaw_vec_x * app_x + yaw_vec_y * app_y
                        sign = 1.0 if dot >= 0 else -1.0
                        
                        dist = random.uniform(20.0, 60.0)
                        perturb_pos[0] += sign * yaw_vec_x * dist
                        perturb_pos[1] += sign * yaw_vec_y * dist
                        mode_info += " | Pos: Overshoot"
                        print(f"  [Perturb] 沿Yaw轴冲过头: {dist:.1f}mm", flush=True)
                    else:
                        noise_x = random.uniform(-self.perturbation_range, self.perturbation_range)
                        noise_y = random.uniform(-self.perturbation_range, self.perturbation_range)
                        perturb_pos[0] += noise_x
                        perturb_pos[1] += noise_y
                        mode_info += " | Pos: Drift"
                        print(f"  [Perturb] 随机漂移: x{noise_x:.1f}, y{noise_y:.1f}", flush=True)

                    perturb_pos[2] = target_pos[2] + random.uniform(25.0, 35.0)
                    perturb_pos[5] = target_pos[5] # Reset Yaw
                    
                    if random.random() < 0.4:
                        yaw_error_deg = random.uniform(20, 45)
                        if random.random() < 0.5: yaw_error_deg *= -1
                        perturb_pos[5] += np.radians(yaw_error_deg)
                        mode_info += f" + Yaw Err {yaw_error_deg:.1f}°"
                        print(f"  [Perturb] >>> 叠加 Yaw 偏差: {yaw_error_deg:.1f}°", flush=True)
                    else:
                        print(f"  [Perturb] >>> 保持正确 Yaw", flush=True)

                    # 执行纠偏动作序列
                    perturb_pos_first = list(perturb_pos)
                    perturb_pos_first[2] += random.uniform(30.0, 40.0)
                    
                    self.move_to(perturb_pos_first, speed=self.speed_fast)
                    self.move_to(perturb_pos, speed=self.speed_fast)
                    time.sleep(1.0)
                    
                    print(f"[Record] 开始录制 Episode {episode_idx} ({mode_info})", flush=True)
                    self.start_recording(episode_idx, self.instruction)
                    
                    target_pos_first = list(target_pos)
                    target_pos_first[2] = perturb_pos_first[2] - random.uniform(10.0, 20.0)
                    self.move_to(target_pos_first, speed=self.speed_adjust) 
                    self.move_to(target_pos, speed=self.speed_adjust) 

                self.close_gripper()
                time.sleep(2.0)
                self.stop_recording()
                print("[Record] 录制完成")

                # 复位
                self.open_gripper() 
                time.sleep(0.5)
                self.move_to(safe_pos_up, speed=self.speed_fast)
                
                episode_idx += 1
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n强制停止...")
            self.stop_recording()
            self.camera_running = False
            self.arm.set_state(4)
            self.cam_thread.join(timeout=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.232", help="xArm IP")
    parser.add_argument("--output", type=str, default="/home/openpi/data/data_raw/exp21_data_auto_queue_PutAndRecord_0115/raw", help="Output directory")
    args = parser.parse_args()
    
    cameras = {
        "cam_left_wrist": "/dev/cam_left_wrist",
        "cam_right_wrist": "/dev/cam_right_wrist"
    }
    
    my_crop_configs = {
        'cam_left_wrist': [118, 60, 357, 420],
        'cam_right_wrist': [136, 57, 349, 412],
    }
    
    recorder = AutoDataRecorder(args.ip, args.output, cameras, crop_configs=my_crop_configs)
    recorder.run_auto_collection()