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
        self.pos_home = [539.120605, 17.047951, 100-69.568863, 3.12897, 0.012689, -1.01436]
        self.pos_A = [539.120605, 17.047951, -69.568863, 3.12897, 0.012689, -1.01436]
        self.fixed_z = self.pos_A[2]  # Z轴固定为桌面高度

        # ===================== 2D凸包区域定义 =====================
        self.boundary_points_2d = np.array([
            [548.982422, -66.848724],  # 左下
            [724.302856, -66.848724],   # 右下
            [724.232117, 240.781003], # 右上
            [428.805481, 203.618057],  # 左上
        ])
        
        # 构建2D凸包
        self.hull_2d = ConvexHull(self.boundary_points_2d)
        self.hull_points_2d = self.boundary_points_2d[self.hull_2d.vertices]
        
        # 计算包围盒（用于快速采样）
        self.min_x = np.min(self.boundary_points_2d[:, 0])
        self.max_x = np.max(self.boundary_points_2d[:, 0])
        self.min_y = np.min(self.boundary_points_2d[:, 1])
        self.max_y = np.max(self.boundary_points_2d[:, 1])
        
        # === 新增：网格分层采样参数 ===
        self.grid_rows = 4  # 将区域分为 4x4 = 16 个格子
        self.grid_cols = 4
        self.total_grids = self.grid_rows * self.grid_cols # 总共9个区
        self.current_grid_idx = 0 # 当前轮到的区域索引 (0-8)
        
        # 这里的列表将存储待访问的格子坐标 [(0,0), (0,1), ... (3,3)]
        self.grid_indices = [] 
        # 初始化并打乱第一次的顺序
        self._refill_grid_indices()
        # ===================== 姿态随机化 =====================
        self.fixed_roll = self.pos_A[3]
        self.fixed_pitch = self.pos_A[4]
        
        # Yaw角度随机范围 (弧度)
        self.base_yaw = self.pos_A[5]
        self.yaw_random_range = (-np.pi/2, np.pi/6)  # ±45度
        self.current_yaw_angle = self.base_yaw

        # 运动速度
        self.speed_fast = 300 
        self.speed_record = 100 
        self.joint_speed_fast = 0.5 
        self.joint_speed_record = 0.15
        
        # === 数据队列 ===
        self.data_queue = queue.Queue(maxsize=50)
        
        # === 扰动参数 ===
        self.perturbation_prob = 0.7  # 70% 的概率加入扰动（故意走偏再修正）
        self.perturbation_range = 30.0 # 扰动范围 30mm (3cm)
        
    def _refill_grid_indices(self):
        """重新填充并打乱网格索引，保证下一轮采集能覆盖所有区域，但顺序随机"""
        self.grid_indices = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                self.grid_indices.append((r, c))
        
        # 核心：打乱列表顺序
        random.shuffle(self.grid_indices) 
        print(f">>> [Sampler] Grid reset & shuffled. Sequence length: {len(self.grid_indices)}")


    # ---------------------------------------------------------
    # 几何算法部分
    # ---------------------------------------------------------
    def is_point_inside_hull_2d(self, point_2d, hull_2d):
        hull_eq = hull_2d.equations
        point_homo = np.hstack([point_2d, 1])
        for eq in hull_eq:
            if np.dot(eq, point_homo) > 1e-6:
                return False
        return True

    def sample_random_in_hull_2d(self):
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            sample_point_2d = np.array([x, y])
            
            if self.is_point_inside_hull_2d(sample_point_2d, self.hull_2d):
                return np.array([x, y, self.fixed_z])
        
        hull_center_2d = np.mean(self.hull_points_2d, axis=0)
        return np.array([hull_center_2d[0], hull_center_2d[1], self.fixed_z])

    # ---------------------------------------------------------
    # 机器人控制
    # ---------------------------------------------------------
    def connect_robot(self):
        if XArmAPI is None: raise ImportError("xArm SDK Missing")
        print(f"Connecting to xArm at {self.ip}...")
        self.arm = XArmAPI(self.ip)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_tgpio_modbus_baudrate(baud=115200)
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
    
    def sample_random_in_grid(self, row, col):
        """
        在指定的 (row, col) 网格内随机采样。
        如果该网格完全在凸包外，可能会返回 None。
        """
        # 计算当前格子的边界
        step_x = (self.max_x - self.min_x) / self.grid_cols
        step_y = (self.max_y - self.min_y) / self.grid_rows
        
        cell_min_x = self.min_x + col * step_x
        cell_max_x = cell_min_x + step_x
        cell_min_y = self.min_y + row * step_y
        cell_max_y = cell_min_y + step_y

        # 在当前格子里尝试采样 (尝试30次)
        # 如果这个格子只有一小角在凸包内，多试几次能命中
        for _ in range(30):
            x = random.uniform(cell_min_x, cell_max_x)
            y = random.uniform(cell_min_y, cell_max_y)
            sample_point_2d = np.array([x, y])
            
            # 必须在凸包(hull)内部
            if self.is_point_inside_hull_2d(sample_point_2d, self.hull_2d):
                return np.array([x, y, self.fixed_z])
        
        return None # 该格子很难采到点（可能在界外）
        
    def sample_random_target(self):
        # 设置一个最大尝试次数，防止极端情况下所有格子都不可达导致死循环
        # 这里设置为总格子数的 2 倍，足够遍历完一轮
        max_checks = self.total_grids * 2
        checks_count = 0
        
        while checks_count < max_checks:
            # 1. 如果列表空了，立刻重新填充并打乱（开启新的一轮循环）
            if not self.grid_indices:
                self._refill_grid_indices()
            
            # 2. 从列表末尾弹出一个格子 (因为列表已经Shuffle过，所以这是随机的)
            r, c = self.grid_indices.pop()
            
            # 用于显示进度的索引（仅作显示用）
            current_idx = r * self.grid_cols + c
            print(f"  [Sampler] Trying Random Grid ({r}, {c}) [ID:{current_idx}]...", end="")

            # 3. 在该格子内尝试生成点
            sample_xyz = self.sample_random_in_grid(r, c)
            
            if sample_xyz is not None:
                target_x, target_y, target_z = sample_xyz
                
                # 4. 尝试匹配一个可行的 Yaw 角度
                for _ in range(10): 
                    yaw_noise = random.uniform(*self.yaw_random_range)
                    candidate_yaw = self.base_yaw + yaw_noise
                    
                    candidate_pose = [
                        target_x, target_y, target_z, 
                        self.fixed_roll, self.fixed_pitch, candidate_yaw
                    ]

                    # 5. 机械臂逆解检查 (可达性)
                    if self.check_pose_reachable(candidate_pose):
                        print(" OK.")
                        self.current_yaw_angle = candidate_yaw
                        return candidate_pose

            # === 如果运行到这里，说明当前随机到的格子不可达或不在凸包内 ===
            print(" Failed/Skip.")
            # 注意：这里不需要手动切到下一个，因为 grid_indices.pop() 已经把它移除列表了
            # 下次循环会自动 pop 列表里的下一个随机格子
            checks_count += 1
        
        # === 保底逻辑 ===
        print("[Error] 无法生成可达点，使用中心保底。")
        region_center_2d = np.mean(self.hull_points_2d, axis=0)
        return [region_center_2d[0], region_center_2d[1], self.fixed_z, 
                self.fixed_roll, self.fixed_pitch, self.base_yaw]

    # ---------------------------------------------------------
    # 相机后台线程 - 核心优化部分
    # ---------------------------------------------------------
    def connect_cameras(self):
        """初始化相机并启动后台采集线程"""
        for name, idx in self.camera_indices.items():
            cap = cv2.VideoCapture(idx)
            # 尽可能请求高FPS，防止硬件缓冲区堆积
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if cap.isOpened():
                self.caps[name] = cap
            else:
                print(f"[Error] 无法打开相机 {name} (Index {idx})")
        
        print(f"Connected to {len(self.caps)} cameras.")
        
        # 启动后台常驻线程
        self.cam_thread = threading.Thread(target=self._camera_daemon, daemon=True)
        self.cam_thread.start()
        
        print(">>> 正在预热相机 (2秒) 以稳定曝光...")
        time.sleep(2.0)

    def _camera_daemon(self):
        """
        后台线程：不间断地读取所有相机。
        作用：清空硬件缓冲区，保证 latest_frames 里永远是此刻的图像。
        """
        while self.camera_running:
            for name, cap in self.caps.items():
                ret, frame = cap.read()
                if ret:
                    with self.frame_lock:
                        self.latest_frames[name] = frame
            # 极短睡眠防止占满CPU，但必须足够快以清空30/60fps的流
            time.sleep(0.002) 

    def _get_latest_images(self):
        """从内存缓存中获取最新图像并裁剪，无IO延迟"""
        images = {}
        with self.frame_lock:
            # 浅拷贝字典，避免处理时发生竞争
            raw_frames = self.latest_frames.copy()
            
        for name, frame in raw_frames.items():
            if name in self.crop_configs and self.crop_configs[name] is not None:
                x, y, w, h = self.crop_configs[name]
                frame = frame[y:y+h, x:x+w]
            images[name] = frame
        return images

    # ---------------------------------------------------------
    # 数据写入线程
    # ---------------------------------------------------------
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
                    
                    # 1. 写 JSON
                    f.write(json.dumps(data_dict) + "\n")
                    
                    # 2. 写 图片
                    frame_idx = data_dict["frame_idx"]
                    for cam_name, img_data in images_dict.items():
                        save_path = episode_dir / "images" / cam_name / f"{frame_idx:06d}.jpg"
                        cv2.imwrite(str(save_path), img_data)
                        
                    self.data_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Writer Error] {e}")

    # ---------------------------------------------------------
    # 录制主线程
    # ---------------------------------------------------------
    def _recording_thread(self, episode_dir, instruction):
        frame_idx = 0
        target_interval = 0.1  # 5Hz
        
        print(f">>> [REC] 启动录制: {episode_dir.name}")
        
        writer = threading.Thread(target=self._writer_worker, args=(episode_dir,))
        writer.start()

        # 获取初始状态
        code, last_joint_pos = self.arm.get_servo_angle(is_radian=True)
        if code != 0: last_joint_pos = [0.0]*7

        while not self.stop_event.is_set():
            loop_start = time.time()
            try:
                # 1. 核心变化：直接从后台线程内存中拿最新图 (0延时)
                current_images = self._get_latest_images()
                
                # 如果刚启动还没拿到图，稍微等一下
                if not current_images and frame_idx == 0:
                    time.sleep(0.01)
                    continue

                # 2. 机器人状态
                code, curr_joint_pos = self.arm.get_servo_angle(is_radian=True)
                ret, curr_cart_pos = self.arm.get_position(is_radian=True)
                
                if ret != 0 or code != 0:
                    time.sleep(0.005); continue
                
                delta_joints = np.array(curr_joint_pos) - np.array(last_joint_pos)
                delta_joints = np.round(delta_joints, 6).tolist()
                
                # 3. 构造数据
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
                
                # 4. 频率控制
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
            
        self.rec_thread = threading.Thread(
            target=self._recording_thread, 
            args=(episode_dir, instruction)
        )
        self.rec_thread.start()

    def stop_recording(self):
        if hasattr(self, 'rec_thread') and self.rec_thread.is_alive():
            self.stop_event.set()
            self.rec_thread.join()

    def input_listener(self):
        while True:
            cmd = input()
            if cmd.strip().lower() == 'p':
                self.pause_requested = True
                print("\n>>> [指令收到] 将在当前循环结束后暂停...")

    # ---------------------------------------------------------
    # 主流程
    # ---------------------------------------------------------
    def run_auto_collection(self):
        self.connect_robot()
        self.connect_cameras() # 这里会启动后台线程
        threading.Thread(target=self.input_listener, daemon=True).start()

        print("\n" + "="*60)
        print("自动数据采集程序 (Zero-Latency Daemon Version)")
        print(f"采样范围：2D凸包 (Vertices: {len(self.boundary_points_2d)})")
        print("特性: 后台线程清空相机缓存，杜绝画面延迟")
        print("操作: 在终端输入 'p' 并回车，可在本轮结束后暂停")
        print("="*60 + "\n")

        # === 阶段 1: 初始化 ===
        print(">>> 初始抓取 (Initial Pickup)...")
        self.move_to(self.pos_A, speed=self.speed_fast)
        self.open_gripper()
        time.sleep(2.0)
        self.move_to(self.pos_home, speed=self.speed_fast)
        self.move_to(self.pos_A, speed=self.speed_fast)
        self.close_gripper()
        time.sleep(2.0)
        self.move_to(self.pos_home, speed=self.speed_fast)

        # === 自动检测 Index ===
        existing_episodes = [
            d.name for d in self.output_dir.iterdir() 
            if d.is_dir() and d.name.startswith("episode_")
        ]
        if existing_episodes:
            indices = []
            for name in existing_episodes:
                try:
                    idx = int(name.split("_")[-1])
                    indices.append(idx)
                except ValueError: continue
            episode_idx = max(indices) + 1 if indices else 0
        else:
            episode_idx = 0
            
        print(f">>> 从 Episode {episode_idx} 开始")
        
        # === 阶段 2: 循环采集 ===
        try:
            while True:
                if self.pause_requested:
                    print("\n" + "!"*40)
                    print(">>> 程序已暂停")
                    input(">>> 按回车键继续...")
                    print("!"*40 + "\n")
                    self.pause_requested = False

                print(f"\n--- Loop {episode_idx} ---")
                
                # 1. 生成随机点 (XYZ + Yaw)
                target_pos = self.sample_random_target()
                safe_pos_up = list(target_pos)
                safe_pos_up[2] += 100 
                safe_pos_up[5] = self.current_yaw_angle # 保持Yaw一致
                
                # 2. [Reset] 放置物体 (不录制)
                print(f"[Reset] 放置到随机点 (Yaw={np.degrees(self.current_yaw_angle):.1f}°)")
                if not self.move_to(safe_pos_up, speed=self.speed_fast): continue
                self.move_to(target_pos, speed=self.speed_fast)
                self.open_gripper()
                time.sleep(2.0)
                self.move_to(safe_pos_up, speed=self.speed_fast)
                self.move_to(self.pos_home, speed=self.speed_fast)
                
                # 3. [Record] 执行抓取 (录制)
                print(f"[Record] 开始录制 Episode {episode_idx}")
                
                time.sleep(1.0) # 确保机械臂完全静止，也确保相机画面稳定
                self.start_recording(episode_idx, "pick up the industrial components")
                
                # 抓取动作
                self.move_to(safe_pos_up, speed=self.speed_record) 
                # --- 核心改进：决定是否加入扰动 ---
                if random.random() < self.perturbation_prob:
                    # 生成一个XY平面的随机偏差 (噪音)
                    noise_x = random.uniform(-self.perturbation_range, self.perturbation_range)
                    noise_y = random.uniform(-self.perturbation_range, self.perturbation_range)
                    
                    # 构造一个“稍微偏一点”的中间点
                    # 高度建议比最终抓取点稍高一点点(比如高1cm)，或者就在同一平面，看你需求
                    # 这里设置为：在抓取点上方 1cm 处，且水平方向有偏差
                    perturb_pos = list(target_pos)
                    perturb_pos[0] += noise_x
                    perturb_pos[1] += noise_y
                    perturb_pos[2] += 10.0 # 稍微高一点，避免直接撞击物体侧面
                    
                    print(f"  [Perturb] 插入扰动: x{noise_x:.1f}, y{noise_y:.1f}")
                    
                    # 动作1: 故意走偏 (移动到扰动点)
                    self.move_to(perturb_pos, speed=self.speed_record)
                    
                    # 动作2: 修正 (从扰动点移动到正确的抓取点)
                    # 这就是模型最需要学习的“纠正行为”！
                    self.move_to(target_pos, speed=self.speed_record * 0.8) # 修正时稍微慢一点，模拟精细操作
                    
                else:
                    # 30% 的概率走完美直线 (作为基准数据)
                    print("  [Direct] 直线抓取")
                    self.move_to(target_pos, speed=self.speed_record)
                self.close_gripper()
                time.sleep(2.0)
                
                self.stop_recording()
                print("[Record] 录制完成")

                # 4. [Return] 复位 (不录制)
                print("[Return] 返回 Home")
                self.move_to(safe_pos_up, speed=self.speed_fast)
                self.move_to_joint_by_pose(self.pos_home, speed=self.speed_fast)
                
                episode_idx += 1
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n强制停止...")
            self.stop_recording()
            self.camera_running = False # 停止相机线程
            self.arm.set_state(4)
            # 等待相机线程结束
            self.cam_thread.join(timeout=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.232", help="xArm IP")
    parser.add_argument("--output", type=str, default="/home/openpi/data/data_raw/exp12_data_auto_queue_PutAndRecord_0104/raw", help="Output directory")
    # parser.add_argument("--output", type=str, default="/home/openpi/data/data_raw/test/raw", help="Output directory")
    args = parser.parse_args()
    
    # cameras = {
    #     "cam_high": 4,
    #     "cam_left_wrist": 0,
    #     "cam_right_wrist": 2
    # }
    cameras = {
        "cam_high": "/dev/cam_high",
        "cam_left_wrist": "/dev/cam_left_wrist",
        "cam_right_wrist": "/dev/cam_right_wrist"
    }
    
    my_crop_configs = {
        'cam_high': [61, 1, 434, 479], 
        'cam_left_wrist': [118, 60, 357, 420],
        'cam_right_wrist': [136, 57, 349, 412],
    }
    
    recorder = AutoDataRecorder(args.ip, args.output, cameras, crop_configs=my_crop_configs)
    recorder.run_auto_collection()