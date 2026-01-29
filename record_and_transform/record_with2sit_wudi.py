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
        self.pos_A = [486.626923, 158.343277, -85.431152, 3.12897, 0.012689, -1.01436]
        # self.instruction = "pick up the industrial components"
        self.instruction =  "pick up the small upright valve"
        self.fixed_z = self.pos_A[2]  # Z轴固定为桌面高度

        # ===================== 2D凸包区域定义 =====================
        self.boundary_points_2d = np.array([
            [514.6, 125.5],
            [706.6, 271.2],
            [475.9, 552.4],
            [337.1,448.0],
        ])
        scale_factor = 0.8  # 缩放比例，0.8 表示缩小到boundary_points_2d的 80%，根据需求调整
        center_point = np.mean(self.boundary_points_2d, axis=0)
        # 将每个顶点向中心点靠拢
        self.boundary_points_2d = center_point + (self.boundary_points_2d - center_point) * scale_factor
        
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
        self.speed_fast = 500 
        self.speed_record = 100 
        self.speed_adjust = 50 
        # self.speed_fast = 500 
        # self.speed_record = 500 
        # self.speed_adjust = 300 
        self.joint_speed_fast = 0.5 
        self.joint_speed_record = 0.15
        
        # === 数据队列 ===
        self.data_queue = queue.Queue(maxsize=50)
        
        # === 扰动参数 ===
        self.perturbation_prob = 0.3  # 60% 的概率加入扰动（故意走偏再修正）
        self.perturbation_range = 30.0 # 扰动范围 30mm (3cm)
       
    def get_random_start_pose(self):
        """
        生成一个随机起始点 (Home)。
        范围：比当前的 boundary_points_2d 更小（向中心收缩），确保安全。
        """
        # 1. 计算中心点
        center = np.mean(self.boundary_points_2d, axis=0)
        
        # 2. 定义缩放系数 (0.8 表示只在中心 80% 的区域内采样 Home 点)
        # 你可以根据需要调整这个值，越小越靠近中心，越大越接近边界
        home_shrink_ratio = 0.8 
        
        # 3. 尝试采样
        for _ in range(50):
            # 在包围盒范围内随机生成 XY
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            point = np.array([x, y])
            
            # --- 核心逻辑：判断点是否在“缩小版”的区域内 ---
            # 我们不需要真的去构建一个新的凸包对象。
            # 数学推导：判断 P 是否在 "缩小 hull" 内，等价于判断 "中心 + (P-中心)/ratio" 是否在 "原 hull" 内。
            # 这一步把点“放大”回去进行检测
            test_point_for_check = center + (point - center) / home_shrink_ratio
            
            if self.is_point_inside_hull_2d(test_point_for_check, self.hull_2d):
                # 4. Z轴高度：Home 点通常比抓取点高
                # 这里设置为 150mm 左右，并带一点随机浮动
                z = self.pos_home[2] + random.uniform(-5.0, 20.0)
                
                # 5. Yaw 角度：保持基础朝向，轻微抖动 (±5度左右)
                # 这样模型能学会处理轻微的初始角度偏差
                yaw = self.base_yaw + random.uniform(-0.1, 0.1)
                
                return [x, y, z, self.fixed_roll, self.fixed_pitch, yaw]
        
        # 保底：如果随机不到，返回固定的 Home 点
        print("[Warn] 随机Home采样失败，使用默认Home")
        return self.pos_home
     
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

    # ... (前面的 __init__ 等保持不变) ...

    def get_next_grid_target(self):
        """
        获取下一个网格目标点。
        逻辑：从 grid_indices 列表里取下一个。如果列表空了，重新 refill。
        """
        if not self.grid_indices:
            self._refill_grid_indices()
        
        # 取出下一个格子的索引
        r, c = self.grid_indices.pop()
        current_idx = r * self.grid_cols + c
        print(f">>> [Target] Switching to Grid ({r}, {c}) [ID:{current_idx}]")

        # 在该格子内随机采样一个点
        # 复用你之前的 sample_random_in_grid 逻辑
        sample_xyz = self.sample_random_in_grid(r, c)
        
        # 如果这个格子采不到点（比如在凸包外），递归找下一个，直到找到为止
        if sample_xyz is None:
            print(f"[Warn] Grid ({r},{c}) invalid, skipping...")
            return self.get_next_grid_target()

        # 生成完整的 6D Pose (XYZ + RPY)
        # Yaw 随机化
        yaw_noise = random.uniform(*self.yaw_random_range)
        candidate_yaw = self.base_yaw + yaw_noise
        
        target_pose = [
            sample_xyz[0], sample_xyz[1], sample_xyz[2], 
            self.fixed_roll, self.fixed_pitch, candidate_yaw
        ]
        
        # 检查可达性，不可达则递归找下一个
        if not self.check_pose_reachable(target_pose):
            print(f"[Warn] Target unreachable, skipping...")
            return self.get_next_grid_target()
            
        self.current_yaw_angle = candidate_yaw
        return target_pose

    # ... (connect_robot, connect_cameras 等保持不变) ...
    
    # ---------------------------------------------------------
    # 机器人控制
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

    # def input_listener(self):
    #     while True:
    #         cmd = input()
    #         if cmd.strip().lower() == 'p':
    #             self.pause_requested = True
    #             print("\n>>> [指令收到] 将在当前循环结束后暂停...")
    
    def input_listener(self):
        """
        监听键盘输入。
        'c' + 回车: 请求切换目标
        'p' + 回车: 请求暂停
        """
        while True:
            try:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == 'c':
                    self.change_target_requested = True
                    print("\n>>> [指令收到] 本轮结束后将切换到下一个目标点...")
                elif cmd == 'p':
                    self.pause_requested = True
                    print("\n>>> [指令收到] 本轮结束后将暂停...")
            except:
                pass

    # ---------------------------------------------------------
    # 主流程
    # ---------------------------------------------------------
    def run_auto_collection(self):
        self.connect_robot()
        self.connect_cameras() # 这里会启动后台线程
        # 初始化标志位
        self.change_target_requested = False
        self.pause_requested = False
        threading.Thread(target=self.input_listener, daemon=True).start()

        print("\n" + "="*60)
        print("自动数据采集程序 (Zero-Latency Daemon Version)")
        print(f"采样范围：2D凸包 (Vertices: {len(self.boundary_points_2d)})")
        print("特性: 后台线程清空相机缓存，杜绝画面延迟")
        print("操作: 在终端输入 'p' 并回车，可在本轮结束后暂停")
        print("="*60 + "\n")

        # 1. 生成第一个目标点
        current_target_pose = self.get_next_grid_target()
        safe_target_up = list(current_target_pose); safe_target_up[2] += 100
        
        # 2. 去 pos_A 抓取
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
        
        print(f">>> [Init] 物体已就位。开始采集循环。")


        
        

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
                # 检查是否需要切换目标点 (Change Target)
                if self.change_target_requested:
                    print("\n" + "="*40)
                    print(">>> [Change] 正在切换到下一个目标点...")
                    # 1. 去当前点抓起物体
                    safe_curr_up = list(current_target_pose); safe_curr_up[2] += 100
                    self.move_to(safe_curr_up, speed=self.speed_fast)
                    self.move_to(current_target_pose, speed=self.speed_fast)
                    self.close_gripper()
                    time.sleep(2.0)
                    self.move_to(safe_curr_up, speed=self.speed_fast)
                    # 2. 生成新目标点
                    new_target_pose = self.get_next_grid_target()
                    safe_new_up = list(new_target_pose); safe_new_up[2] += 100
                    
                    # 3. 移动并放置
                    # self.move_to(self.pos_home, speed=self.speed_fast) # 中途回个家，防缠绕
                    self.move_to(safe_new_up, speed=self.speed_fast)
                    self.move_to(new_target_pose, speed=self.speed_fast)
                    self.open_gripper()
                    time.sleep(2.0)
                    self.move_to(safe_new_up, speed=self.speed_fast)
                    
                    # 4. 更新状态
                    current_target_pose = new_target_pose
                    self.change_target_requested = False
                    print(">>> [Change] 切换完成。继续采集。")
                    print("="*40 + "\n")
                print(f"\n--- Loop {episode_idx} (Target: {self.current_grid_idx}) ---")
                
                # 采集流程 (Random Home -> Fixed Target)
                # 1. 生成本轮的随机起始点
                current_home_pos = self.get_random_start_pose()
                
                # 2. 移动到随机起始点 (不录制)
                self.move_to(safe_target_up, speed=self.speed_fast) # 先抬高，防撞
                self.move_to_joint_by_pose(current_home_pos, speed=self.speed_fast)
                time.sleep(0.5)
                # 3. 决定采集模式 (全流程 / 纠偏)
                # 复用你之前的逻辑，只是 target_pos 现在是固定的 current_target_pose
                
                target_pos = current_target_pose # 使用当前固定的点
                
                # ==========================================
                # 【新增】随机化本轮的起始点 (Random Home)
                # ==========================================
                # current_home_pos = self.get_random_start_pose()
                print(f"[Home] 随机起始点: ({current_home_pos[0]:.1f}, {current_home_pos[1]:.1f}, {current_home_pos[2]:.1f})")
                
                # 1. 生成随机点 (XYZ + Yaw)
                # target_pos = self.sample_random_target()
                safe_pos_up = list(target_pos)
                safe_pos_up[2] += 100 
                safe_pos_up[5] = self.current_yaw_angle # 保持Yaw一致
                
                # # 2. [Reset] 放置物体 (不录制)
                # print(f"[Reset] 放置到随机点 (Yaw={np.degrees(self.current_yaw_angle):.1f}°)")
                # if not self.move_to(safe_pos_up, speed=self.speed_fast): continue
                # self.move_to(target_pos, speed=self.speed_fast)
                # self.open_gripper()
                # time.sleep(2.0)
                # self.move_to(safe_pos_up, speed=self.speed_fast)
                
                # #放置完物体后，不回固定 Home，而是去随机 Home
                # self.move_to(current_home_pos, speed=self.speed_fast)
                
                # 3. [Record] 执行抓取 (录制)
    
                # [Record] 混合采集策略.
                if random.random() > self.perturbation_prob:
                # === 模式 A: 标准全流程 (Home -> Up -> Target) ===
                    print("[Prepare] 模式: 标准全流程 (Standard Approach)")
                    # 1. 确保在 Home 点
                    # self.move_to(self.pos_home, speed=self.speed_fast)
                    time.sleep(1.0)
                    # 2. 【从 Home 就开始录像】
                    print(f"[Record] 开始录制 Episode {episode_idx} (Full)")
                    self.start_recording(episode_idx, self.instruction)
                    # 3. 执行动作：先到上方，再下去
                    # 这里必须用 speed_record (慢速)，因为这是要学习的动作
                    self.move_to(safe_pos_up, speed=self.speed_record)
                    self.move_to(target_pos, speed=self.speed_record)
                
                else:
                 
                # --- 核心改进：决定是否加入扰动 ---
                # if random.random() < self.perturbation_prob:
                    # # 生成一个XY平面的随机偏差 (噪音)
                    # noise_x = random.uniform(-self.perturbation_range, self.perturbation_range)
                    # noise_y = random.uniform(-self.perturbation_range, self.perturbation_range)
                    
                    # # 构造一个“稍微偏一点”的中间点
                    # # 高度建议比最终抓取点稍高一点点(比如高1cm)，或者就在同一平面，看你需求
                    # # 这里设置为：在抓取点上方 1cm 处，且水平方向有偏差
                    # perturb_pos = list(target_pos)
                    # perturb_pos[0] += noise_x
                    # perturb_pos[1] += noise_y
                    # perturb_pos[2] += 10.0 # 稍微高一点，避免直接撞击物体侧面
                    
                    # print(f"  [Perturb] 插入扰动: x{noise_x:.1f}, y{noise_y:.1f}")
                    
                    # # 动作1: 故意走偏 (移动到扰动点)
                    # self.move_to(perturb_pos, speed=self.speed_record)
                    
                    # # 动作2: 修正 (从扰动点移动到正确的抓取点)
                    # # 这就是模型最需要学习的“纠正行为”！
                    # self.move_to(target_pos, speed=self.speed_record * 0.8) # 修正时稍微慢一点，模拟精细操作
                    
                    
                    #用于模拟走过头，而非原先的上方随机点
                    # 计算“冲过头”的方向向量
                    # 向量 = 目标点(XY) - 起始点(XY)
                    vec_x = target_pos[0] - safe_pos_up[0]
                    vec_y = target_pos[1] - safe_pos_up[1]
                    
                    # 归一化向量
                    norm = np.sqrt(vec_x**2 + vec_y**2)
                    if norm > 0:
                        dir_x = vec_x / norm
                        dir_y = vec_y / norm
                    else:
                        dir_x, dir_y = 0, 0
                    # 定义扰动类型
                    # 70% 的概率模拟“冲过头”（沿着运动方向继续往前冲）
                    # 30% 的概率模拟“左右偏”（随机噪音）
                    if random.random() < 0.7:
                        print(f"  [Perturb] 模拟沿着yaw偏差")
                        # 1. 获取当前的 Yaw 角度
                        target_yaw = target_pos[5]
                        # 2. 计算 Yaw 角度对应的单位向量 (夹爪的朝向线)
                        yaw_vec_x = np.cos(target_yaw)
                        yaw_vec_y = np.sin(target_yaw)
                        # 3. 判断哪个方向是“远离起始点”的
                        # 计算从 Home 到 Target 的宏观趋势向量
                        approach_vec_x = target_pos[0] - self.pos_home[0]
                        approach_vec_y = target_pos[1] - self.pos_home[1]
                        # 使用点积判断方向一致性
                        # dot > 0: Yaw指向与宏观运动方向大致相同 -> 沿 Yaw 正方向冲
                        # dot < 0: Yaw指向与宏观运动方向相反 -> 沿 Yaw 负方向冲
                        dot_product = yaw_vec_x * approach_vec_x + yaw_vec_y * approach_vec_y
                        direction_sign = 1.0 if dot_product >= 0 else -1.0
                        # 4. 定义扰动参数
                        # 冲过头的距离：3cm 到 6cm
                        overshoot_dist = random.uniform(20.0, 60.0)
                        # 构造扰动点 (错误点)
                        perturb_pos = list(target_pos)
                        # -> 沿着 Yaw 确定的直线，往远离 Home 的方向移动
                        perturb_pos[0] += direction_sign * yaw_vec_x * overshoot_dist
                        perturb_pos[1] += direction_sign * yaw_vec_y * overshoot_dist
                        # -> Z 轴处理：同样保持“斜向下插”的趋势
                        # 比最终点高 1.5cm ~ 3cm，模拟没刹住车斜着滑过去
                        # perturb_pos[2] = target_pos[2] + random.uniform(15.0, 30.0)
                        perturb_pos[2] = target_pos[2] + random.uniform(25.0, 35.0)
                        # -> Yaw 严格保持不变 (这是关键，姿态是准的，只是位置滑了)
                        perturb_pos[5] = target_yaw
                        
                        print(f"  [Perturb] 沿Yaw轴冲过头: dist={overshoot_dist:.1f}mm, Yaw={np.degrees(target_yaw):.1f}°")
                        print(perturb_pos)
                        if random.random() < 0.8:
                            yaw_error_deg = random.uniform(20, 45)
                            if random.random() < 0.5: yaw_error_deg *= -1
                            perturb_pos[5] += np.radians(yaw_error_deg)
                            # mode_info += f" + Yaw Err {yaw_error_deg:.1f}°"
                            print(f"  [Perturb] >>> 触发 Yaw 偏差: {yaw_error_deg:.1f}°", flush=True)
                        else:
                            # 保持正确的 Yaw (只偏位置)
                            perturb_pos[5] = target_pos[5]
                            print("保持正确的yaw")
                            print(f"  [Perturb] >>> 保持正确 Yaw (Random val: {random.random():.2f})", flush=True)
                        print(perturb_pos)
                        # === 执行动作序列 ===
                        perturb_pos_first = list(perturb_pos)
                        perturb_pos_first[2] = target_pos[2] + random.uniform(90.0, 110.0)
                        self.move_to(perturb_pos_first, speed=self.speed_fast)
                        # 动作 1: 带着惯性，沿着夹爪朝向冲过头
                        # self.move_to(perturb_pos, speed=self.speed_fast)
                        time.sleep(1.0)
                        print(f"[Record] 开始录制 Episode {episode_idx} (Correction)")
                        self.start_recording(episode_idx, self.instruction)
                        
                        # 动作 2: 修正 (回拉并下降)
                        # 这一步教会模型：当发现自己沿着夹爪方向跑过了，要倒车回来
                        target_pos_first = list(target_pos)
                        target_pos_first[2] = perturb_pos_first[2] - random.uniform(10.0, 20.0)
                        self.move_to(target_pos_first, speed=self.speed_adjust)
                        self.move_to(target_pos, speed=self.speed_adjust)
                        
                       
                    else:
                        # 模拟随机偏离（原来的逻辑）
                        noise_x = random.uniform(-self.perturbation_range, self.perturbation_range)
                        noise_y = random.uniform(-self.perturbation_range, self.perturbation_range)
                        print(f"  [Perturb] 模拟随机偏差 (Random Drift): x{noise_x:.1f}, y{noise_y:.1f}")
                    
                        # 构造扰动点（错误点）
                        perturb_pos = list(target_pos)
                        perturb_pos[0] += noise_x
                        perturb_pos[1] += noise_y
                        # 关键：扰动点的高度要稍微高一点 (比如+15mm)，防止冲过头时把物体撞飞
                        # perturb_pos[2] += 25.0 #工件C
                        perturb_pos[2] += 45.0 
                        
                        print(perturb_pos)
                        if random.random() < 0.8:
                            yaw_error_deg = random.uniform(20, 45)
                            if random.random() < 0.5: yaw_error_deg *= -1
                            perturb_pos[5] += np.radians(yaw_error_deg)
                            # mode_info += f" + Yaw Err {yaw_error_deg:.1f}°"
                            print(f"  [Perturb] >>> 触发 Yaw 偏差: {yaw_error_deg:.1f}°", flush=True)
                        else:
                            # 保持正确的 Yaw (只偏位置)
                            perturb_pos[5] = target_pos[5]
                            print("保持正确的yaw")
                            print(f"  [Perturb] >>> 保持正确 Yaw (Random val: {random.random():.2f})", flush=True)
                        print(perturb_pos)

                        perturb_pos_first = list(perturb_pos)
                        perturb_pos_first[2] += random.uniform(30.0, 40.0)
                        self.move_to(perturb_pos_first, speed=self.speed_fast)
                        # 动作1: 执行“错误”的移动 (移动到扰动点)
                        # self.move_to(perturb_pos, speed=self.speed_fast)
                        time.sleep(1.0)
                        print(f"[Record] 开始录制 Episode {episode_idx} (Correction)")
                        self.start_recording(episode_idx, self.instruction)
                        
                        # 动作2: 执行“修正” (从扰动点拉回到正确的抓取点)
                        # 稍微降速，让模型看清楚“我是怎么回来的”
                        target_pos_first = list(target_pos)
                        target_pos_first[2] = perturb_pos_first[2] - random.uniform(10.0, 20.0)
                        self.move_to(target_pos_first, speed=self.speed_adjust) 
                        self.move_to(target_pos, speed=self.speed_adjust) 
                    
                
                self.close_gripper()
                time.sleep(2.0)
                
                self.stop_recording()
                print("[Record] 录制完成")

                # 5. 复位 (把物体放回原处)
                # 注意：因为我们是“固定目标点采集”，抓起来后得放回去，才能进行下一次采集
                # 除非你想每次都抓起来然后扔掉？通常是原地放下。
                self.open_gripper() 
                time.sleep(2.0)
                # 抬起，准备下一轮
                safe_pos_up = list(target_pos); safe_pos_up[2] += 100
                self.move_to(safe_pos_up, speed=self.speed_fast)
                
               
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
    parser.add_argument("--output", type=str, default="/home/openpi/data/data_raw/exp21_data_auto_queue_PutAndRecord_0115/raw", help="Output directory")
    # parser.add_argument("--output", type=str, default="/home/openpi/data/data_raw/test/raw", help="Output directory")
    args = parser.parse_args()
    
    # cameras = {
    #     "cam_high": 4,
    #     "cam_left_wrist": 0,
    #     "cam_right_wrist": 2
    # }
    cameras = {
        "cam_left_wrist": "/dev/cam_left_wrist",
        "cam_right_wrist": "/dev/cam_right_wrist"
    }
    
    my_crop_configs = {
        # 'cam_high': [61, 1, 434, 479], 
        'cam_left_wrist': [118, 60, 357, 420],
        'cam_right_wrist': [136, 57, 349, 412],
    }
    
    recorder = AutoDataRecorder(args.ip, args.output, cameras, crop_configs=my_crop_configs)
    recorder.run_auto_collection()