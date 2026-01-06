# import time
# import cv2
# import numpy as np
# import sys
# import os
# import termios
# import select
# import random
# from pathlib import Path
# from scipy.spatial import ConvexHull

# from xarm.wrapper import XArmAPI

# # -----------------------------------------------------------------------------
# # 路径设置
# # -----------------------------------------------------------------------------
# current_dir = os.path.dirname(os.path.abspath(__file__))
# openpi_client_path = os.path.join(current_dir, "../../../packages/openpi-client/src")
# sys.path.append("/home/openpi/src")
# sys.path.append(os.path.abspath(openpi_client_path))

# from openpi.training import config as _config
# from openpi.policies import policy_config
# from openpi_client import image_tools

# # -----------------------------------------------------------------------------
# # 1. 配置区域
# # -----------------------------------------------------------------------------
# ROBOT_IP = "192.168.1.232"
# CONFIG_NAME = "pi05_xarm_1212_night" 
# CHECKPOINT_DIR = "/home/openpi/checkpoints/exp9/94000"
# # 可视化保存路径
# VIS_SAVE_DIR = "/home/openpi/examples/xarm_real/images"

# CAMERAS = {
#     "cam_left_wrist": 0,
#     "cam_right_wrist": 2
# }
# CROP_CONFIGS = {
#     "cam_left_wrist": (118, 60, 357, 420),
#     "cam_right_wrist": (136, 57, 349, 412)
# }

# CONTROL_FREQ = 10 
# EXECUTE_STEPS = 1 
# JOINT_LIMITS = [
#     (-6.2, 6.2), (-2.0, 2.0), (-2.9, 2.9), 
#     (-3.1, 3.1), (-1.6, 1.8), (-6.2, 6.2)
# ]

# HOME_POS = [539.120605, 17.047951, 100-59.568863, 3.12897, 0.012689, -1.01436]
# POS_A = [539.120605, 17.047951, -69.568863, 3.12897, 0.012689, -1.01436]
# MIN_SAFE_Z = -69
# SLOW_DOWN_FACTOR = 3.0  
# INTERPOLATION_FREQ = 100.0 

# # 2D凸包区域定义
# # BOUNDARY_POINTS_2D = np.array([
# #     [505.982422, -150.631149],  # 左下
# #     [724.302856, -66.848724],   # 右下
# #     [724.232117, 240.781003],   # 右上
# #     [428.805481, 203.618057],   # 左上
# # ])

# #exp9
# BOUNDARY_POINTS_2D = np.array([
#            [505.982422, -150.631149],  # 左下
#            [712.302856, -66.848724],   # 右下
#            [697.232117, 163.981003], # 右上
#            [466.805481, 144.618057],  # 左上
#        ])

# FIXED_Z = POS_A[2]
# FIXED_ROLL = POS_A[3]
# FIXED_PITCH = POS_A[4]
# BASE_YAW = POS_A[5]
# YAW_RANDOM_RANGE = (-np.pi/6, np.pi/6)

# # -----------------------------------------------------------------------------
# # 2. 工具类：可视化量化器
# # -----------------------------------------------------------------------------
# class TaskVisualizer:
#     def __init__(self, save_dir, boundary_points, home_pos):
#         self.save_dir = Path(save_dir)
#         self.save_dir.mkdir(parents=True, exist_ok=True)
#         self.save_path = self.save_dir / "performance_map_exp9_94000_0105rainingDay.png"
#         self.boundary = boundary_points
#         self.home_pos = home_pos
        
#         # 坐标映射参数
#         pad = 50
#         min_x, max_x = np.min(boundary_points[:, 0]), np.max(boundary_points[:, 0])
#         min_y, max_y = np.min(boundary_points[:, 1]), np.max(boundary_points[:, 1])
        
#         self.scale = 1.5 # 1mm = 1.5 pixels
#         self.offset_x = -min_x * self.scale + pad
#         self.offset_y = -min_y * self.scale + pad
        
#         width = int((max_x - min_x) * self.scale + pad * 2)
#         height = int((max_y - min_y) * self.scale + pad * 2)
#         self.img_h, self.img_w = height, width

#         self.canvas = self._load_or_create()

#     def _to_pixel(self, x, y):
#         u = int(x * self.scale + self.offset_x)
#         v = int(y * self.scale + self.offset_y)
#         return (u, v)

#     def _load_or_create(self):
#         img = None
#         # 尝试加载
#         if self.save_path.exists():
#             img = cv2.imread(str(self.save_path))
        
#         # 如果加载失败或不存在
#         if img is None:
#             print("图像为空或者目录下没有此图") # 按要求输出提示
            
#             # 创建白底新图
#             img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
            
#             # 画不规则四边形轮廓 (黑色线条)
#             pts = []
#             for p in self.boundary:
#                 pts.append(self._to_pixel(p[0], p[1]))
#             pts = np.array(pts, np.int32)
#             pts = pts.reshape((-1, 1, 2))
            
#             # 连线
#             cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)
            
#             # 画四个角点
#             for p in pts:
#                 cv2.circle(img, (p[0][0], p[0][1]), 6, (0, 0, 0), -1)
                
#             hx, hy = self.home_pos[0], self.home_pos[1]
#             hu, hv = self._to_pixel(hx, hy)
#             # 画蓝色实心圆 (BGR: 255, 0, 0)
#             cv2.circle(img, (hu, hv), 8, (255, 0, 0), -1)
                
#             cv2.imwrite(str(self.save_path), img)
#             print(f"[Vis] Created new map at {self.save_path}")
#         else:
#             print(f"[Vis] Loaded existing map: {self.save_path}")
            
#         return img

#     def update_result(self, pose, success):
#         """
#         pose: [x, y, z, r, p, yaw]
#         success: Bool
#         """
#         x, y, yaw = pose[0], pose[1], pose[5]
#         start_pt = self._to_pixel(x, y)
        
#         # 颜色: BGR. 成功=绿(0,255,0), 失败=红(0,0,255)
#         color = (0, 255, 0) if success else (0, 0, 255)
        
#         # 计算箭头终点 (长度 25 像素)
#         arrow_len_px = 25
#         # 注意：这里需要在图像空间计算方向。
#         # 由于我们采用的是 x->u, y->v 的线性缩放，角度计算如下：
#         end_x_px = start_pt[0] + arrow_len_px * np.cos(yaw)
#         end_y_px = start_pt[1] + arrow_len_px * np.sin(yaw)
#         end_pt = (int(end_x_px), int(end_y_px))
        
#         # 画点
#         cv2.circle(self.canvas, start_pt, 4, color, -1) 
        
#         # 画箭头 (表示 Yaw 朝向)
#         cv2.arrowedLine(self.canvas, start_pt, end_pt, color, thickness=2, tipLength=0.3)
        
#         # 保存
#         cv2.imwrite(str(self.save_path), self.canvas)
#         print(f"[Vis] Map updated (Yaw arrow added).")

# # -----------------------------------------------------------------------------
# # 3. 工具类：采样器
# # -----------------------------------------------------------------------------
# class TaskSampler:
#     def __init__(self):
#         self.hull_2d = ConvexHull(BOUNDARY_POINTS_2D)
#         self.boundary_points = BOUNDARY_POINTS_2D
#         self.min_x = np.min(BOUNDARY_POINTS_2D[:, 0])
#         self.max_x = np.max(BOUNDARY_POINTS_2D[:, 0])
#         self.min_y = np.min(BOUNDARY_POINTS_2D[:, 1])
#         self.max_y = np.max(BOUNDARY_POINTS_2D[:, 1])

#         self.grid_rows = 4
#         self.grid_cols = 4
#         self.grid_indices = []
#         self._refill_grid_indices()

#         ccw_indices = self.hull_2d.vertices
#         self.ccw_vertices = BOUNDARY_POINTS_2D[ccw_indices]
#         self.boundary_step_size = 20.0
#         self.path_points_2d = self._generate_boundary_path(self.ccw_vertices, self.boundary_step_size)
#         self.total_path_points = len(self.path_points_2d)
        
#     def _refill_grid_indices(self):
#         self.grid_indices = []
#         for r in range(self.grid_rows):
#             for c in range(self.grid_cols):
#                 self.grid_indices.append((r, c))
#         random.shuffle(self.grid_indices)

#     def _generate_boundary_path(self, vertices, step_size):
#         path = []
#         num_v = len(vertices)
#         for i in range(num_v):
#             p_curr = vertices[i]
#             p_next = vertices[(i + 1) % num_v]
#             vec = p_next - p_curr
#             dist = np.linalg.norm(vec)
#             steps = int(max(1, dist / step_size))
#             unit_vec = vec / dist
#             for s in range(steps):
#                 point = p_curr + unit_vec * (s * step_size)
#                 path.append(point)
#         return np.array(path)

#     def is_inside(self, x, y):
#         point_homo = np.array([x, y, 1])
#         for eq in self.hull_2d.equations:
#             if np.dot(eq, point_homo) > 1e-6: return False
#         return True

#     def get_random_grid_target(self):
#         for _ in range(self.grid_rows * self.grid_cols * 2):
#             if not self.grid_indices: self._refill_grid_indices()
#             r, c = self.grid_indices.pop()
#             step_x = (self.max_x - self.min_x) / self.grid_cols
#             step_y = (self.max_y - self.min_y) / self.grid_rows
#             min_x, max_x = self.min_x + c * step_x, self.min_x + (c+1) * step_x
#             min_y, max_y = self.min_y + r * step_y, self.min_y + (r+1) * step_y

#             for _ in range(30):
#                 tx = random.uniform(min_x, max_x)
#                 ty = random.uniform(min_y, max_y)
#                 if self.is_inside(tx, ty):
#                     yaw = BASE_YAW + random.uniform(*YAW_RANDOM_RANGE)
#                     return [tx, ty, FIXED_Z, FIXED_ROLL, FIXED_PITCH, yaw]
#         c = np.mean(self.boundary_points, axis=0)
#         return [c[0], c[1], FIXED_Z, FIXED_ROLL, FIXED_PITCH, BASE_YAW]

#     def get_boundary_target(self):
#         rand_idx = random.randint(0, self.total_path_points - 1)
#         pt = self.path_points_2d[rand_idx]
#         yaw = BASE_YAW + random.uniform(*YAW_RANDOM_RANGE)
#         return [pt[0], pt[1], FIXED_Z, FIXED_ROLL, FIXED_PITCH, yaw]

# # -----------------------------------------------------------------------------
# # 4. 工具类：非阻塞键盘检测
# # -----------------------------------------------------------------------------
# class KeyPoller:
#     def __enter__(self):
#         self.fd = sys.stdin.fileno()
#         self.old_term = termios.tcgetattr(self.fd)
#         new_term = termios.tcgetattr(self.fd)
#         new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)
#         termios.tcsetattr(self.fd, termios.TCSANOW, new_term)
#         return self
#     def __exit__(self, type, value, traceback):
#         termios.tcsetattr(self.fd, termios.TCSANOW, self.old_term)
#     def poll(self):
#         dr, dw, de = select.select([sys.stdin], [], [], 0)
#         if not dr: return None
#         return sys.stdin.read(1)

# # -----------------------------------------------------------------------------
# # 5. 硬件封装类
# # -----------------------------------------------------------------------------
# class XArmHardware:
#     def __init__(self, ip, camera_indices):
#         print(f"Connecting to xArm at {ip}...")
#         self.arm = XArmAPI(ip)
#         self.arm.clean_error()
#         self.arm.clean_warn()
#         self.arm.motion_enable(enable=True)
#         self.arm.set_tgpio_modbus_baudrate(baud=115200)
        
        
#         self.caps = {}
#         for name, idx in camera_indices.items():
#             cap = cv2.VideoCapture(idx)
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#             cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#             if cap.isOpened():
#                 self.caps[name] = cap
#             else:
#                 print(f"[Warn] Failed to open camera {name}")
        
#         self.current_gripper_state = 0.0
#         self.arm.set_mode(0)
#         self.arm.set_state(0)
#         self.open_gripper()
#         time.sleep(1.0) 

#     def close_gripper(self):
#         self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
#         self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
#         self.current_gripper_state = 1.0

#     def open_gripper(self):
#         self.arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])
#         self.arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
#         self.current_gripper_state = 0.0

#     def get_observation(self) -> dict:
#         obs = {}
#         for name, cap in self.caps.items():
#             ret, frame = cap.read()
#             if not ret: frame = np.zeros((480, 640, 3), dtype=np.uint8)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             if name in CROP_CONFIGS:
#                 x, y, w, h = CROP_CONFIGS[name]
#                 frame = frame[y:y+h, x:x+w]
#             obs[name] = image_tools.convert_to_uint8(frame)
#             # print(f"[{name}] 图片尺寸: {frame.shape}")
#         dummy_high = np.zeros((224, 224, 3), dtype=np.uint8) 
#         obs["cam_high"] = dummy_high
#         code, joints_rad = self.arm.get_servo_angle(is_radian=True)
#         if code != 0: 
#             print(f"[Warn] Get joint error: {code}")
#             joints_rad = [0.0] * 7
        
#         state_vec = np.array(joints_rad[:6], dtype=np.float32)
#         state_vec_final = np.append(state_vec, self.current_gripper_state)
#         obs["state"] = state_vec_final
#         return obs

#     def move_to_pose_scripted(self, pose, speed=100, wait=True):
#         self.arm.set_mode(0); self.arm.set_state(0)
#         self.arm.set_position(
#             x=pose[0], y=pose[1], z=pose[2],
#             roll=pose[3], pitch=pose[4], yaw=pose[5],
#             speed=speed, wait=wait, is_radian=True
#         )

#     def move_home_scripted(self):
#         print(">>> Moving Home...")
#         self.move_to_pose_scripted(HOME_POS, speed=150)

#     def run_setup_sequence(self, target_pose):
#         print(">>> Executing Setup Sequence...")
#         speed_fast = 300
#         speed_precise = 150
#         pose_A_up = list(POS_A); pose_A_up[2] += 100
#         target_up = list(target_pose); target_up[2] += 100

#         try:
#             # Home -> A(Pick) -> Target(Place) -> Home
#             self.move_to_pose_scripted(pose_A_up, speed=speed_fast)
#             self.move_to_pose_scripted(POS_A, speed=speed_precise)
#             self.close_gripper()
#             time.sleep(0.8)
            
#             self.move_to_pose_scripted(pose_A_up, speed=speed_fast)
#             self.move_home_scripted()
            
#             self.move_to_pose_scripted(target_up, speed=speed_fast)
#             self.move_to_pose_scripted(target_pose, speed=speed_precise)
#             self.open_gripper()
#             time.sleep(1.0)
            
#             self.move_to_pose_scripted(target_up, speed=speed_fast)
#             self.move_home_scripted()
#             print(">>> Sequence Done.")
#         except Exception as e:
#             print(f"[Error] Setup failed: {e}")
#             self.move_home_scripted()

#     def move_to_start(self, target_action_rad):
#         print(">>> Moving to inference start position...")
#         target_joints = target_action_rad[:6]
#         safe_joints = [np.clip(angle, low, high) for angle, (low, high) in zip(target_joints, JOINT_LIMITS)]
#         self.arm.set_mode(0); self.arm.set_state(0)
#         self.arm.set_servo_angle(angle=safe_joints, speed=0.35, is_radian=True, wait=True)
#         self.arm.set_mode(1); self.arm.set_state(0)
#         time.sleep(0.5)

#     def execute_action(self, action_rad):
#         target_joints_rad = action_rad[:6]
#         target_gripper = action_rad[6]
#         safe_joints = [np.clip(angle, low, high) for angle, (low, high) in zip(target_joints_rad, JOINT_LIMITS)]
        
#         code, current_joints = self.arm.get_servo_angle(is_radian=True)
#         if code != 0:
#             print("[Error] Failed to get current joints, skip interpolation.")
#             return
#         curr_j = np.array(current_joints[:6])
#         targ_j = np.array(safe_joints[:6])
        
#         duration = (1.0 / CONTROL_FREQ) * SLOW_DOWN_FACTOR
#         steps = int(duration * INTERPOLATION_FREQ)
#         if steps < 1: steps = 1
        
#         for i in range(1, steps + 1):
#             alpha = i / steps
#             interp = curr_j + (targ_j - curr_j) * alpha
#             full = np.append(interp, 0.0)
#             self.arm.set_servo_angle_j(angles=full, is_radian=True)
#             time.sleep(1.0 / INTERPOLATION_FREQ)

#         if target_gripper > 0.8: self.close_gripper()
#         elif target_gripper < 0.2: self.open_gripper()

#     def close(self):
#         self.arm.set_mode(0); self.arm.set_state(0)
#         for cap in self.caps.values(): cap.release()
#         self.arm.disconnect()

# # -----------------------------------------------------------------------------
# # 6. 主程序逻辑
# # -----------------------------------------------------------------------------
# def main():
#     print(f"Loading Config & Model...")
#     config = _config.get_config(CONFIG_NAME)
#     policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
#     print("Model Loaded.")

#     robot = XArmHardware(ROBOT_IP, CAMERAS)
#     sampler = TaskSampler()
#     # 初始化可视化工具
#     viz = TaskVisualizer(VIS_SAVE_DIR, BOUNDARY_POINTS_2D, HOME_POS)
    
#     prompt = "pick up the industrial components"
#     current_target_pose = None
    
#     try:
#         episode_count = 0
#         while True:
#             episode_count += 1
#             print(f"\n" + "="*50)
#             print(f"Episode {episode_count} | Visualization active")
#             print(f"Controls:")
#             print(f"  [r] : Grid Place (Set target)")
#             print(f"  [b] : Boundary Place (Set target)")
#             print(f"  [ENTER] : Run Inference")
#             print(f"  [o] : Reset Robot (Immediate stop)")
#             print("="*50)

#             print(">>> Waiting for input...", end="", flush=True)
#             mode = None 
            
#             # --- 键盘监听 (设置阶段) ---
#             with KeyPoller() as key_poller:
#                 while True:
#                     key = key_poller.poll()
#                     if key is not None:
#                         if key == '\n' or key == '\r': # Enter
#                             mode = 'inference'
#                             print("\n>>> Inference Started.")
#                             break
#                         elif key == 'r':
#                             print("\n>>> Random Grid Sampling...")
#                             current_target_pose = sampler.get_random_grid_target()
#                             robot.run_setup_sequence(current_target_pose)
#                             print(">>> Setup Done. Press ENTER to run model.")
#                         elif key == 'b': 
#                             print("\n>>> Boundary Sampling...")
#                             current_target_pose = sampler.get_boundary_target()
#                             robot.run_setup_sequence(current_target_pose)
#                             print(">>> Setup Done. Press ENTER to run model.")
#                         elif key == 'o': 
#                             print("\n>>> Resetting...")
#                             robot.open_gripper()
#                             robot.move_home_scripted()
#                             print(">>> Reset Done.")
#                         elif key == '\x03':
#                             raise KeyboardInterrupt
#                     time.sleep(0.01)

#             if mode != 'inference': continue

#             # === 推理流程 ===
#             print("[Phase 1] Aligning...")
#             with KeyPoller() as key_poller:
#                 if key_poller.poll() == 'o':
#                     robot.open_gripper(); robot.move_home_scripted()
#                     continue

#             raw_obs = robot.get_observation()
#             example = {
#                 "cam_left_wrist": raw_obs["cam_left_wrist"],
#                 "cam_right_wrist": raw_obs["cam_right_wrist"],
#                 "state": raw_obs["state"],
#                 "prompt": prompt
#             }
#             result = policy.infer(example)
#             start_action_rad = np.array(result["actions"])[0]
#             robot.move_to_start(start_action_rad)

#             print("[Phase 2] AI Loop (Press 'o' to abort & evaluate)...")
#             aborted = False

#             with KeyPoller() as key_poller:
#                 while True:
#                     # 如果按下 'o'，中断循环，但会进入下方的评估流程
#                     if key_poller.poll() == 'o':
#                         print("\n>>> ABORTED by user (Emergency Stop).")
#                         aborted = True
#                         break
                    
#                     raw_obs = robot.get_observation()
#                     example = {
#                         "cam_left_wrist": raw_obs["cam_left_wrist"],
#                         "cam_right_wrist": raw_obs["cam_right_wrist"],
#                         "state": raw_obs["state"],
#                         "prompt": prompt
#                     }

#                     result = policy.infer(example)
#                     action_chunk = np.array(result["actions"]) 

#                     # 自动检测抓取
#                     if np.any(action_chunk[:1, 6] > 0.8):
#                         print(f"\n>>> Grasp detected (Auto Stop).")
#                         break

#                     steps_to_run = min(EXECUTE_STEPS, len(action_chunk))
#                     for i in range(steps_to_run):
#                         st = time.time()
#                         if key_poller.poll() == 'o': aborted = True; break
#                         robot.execute_action(action_chunk[i])
#                         el = time.time() - st
#                         if (1.0/CONTROL_FREQ - el) > 0: time.sleep(1.0/CONTROL_FREQ - el)
                    
#                     if aborted: break

#             # =========================================================
#             # 结果确认环节 (无论正常结束还是 'o' 键中断，只要有目标点就问)
#             # =========================================================
#             if current_target_pose is not None:
#                 print("\n" + "*"*50)
#                 print(f"EVALUATION REQUIRED for Target: ({current_target_pose[0]:.1f}, {current_target_pose[1]:.1f})")
#                 print("Was the grasp successful?")
#                 print("  [y] : Yes -> Mark Green Arrow")
#                 print("  [n] : No  -> Mark Red Arrow")
#                 print("*"*50)
                
#                 valid_input = False
#                 # 清除输入缓冲
#                 termios.tcflush(sys.stdin, termios.TCIOFLUSH)
                
#                 with KeyPoller() as key_poller:
#                     while not valid_input:
#                         key = key_poller.poll()
#                         if key == 'y':
#                             print(">>> Marked as SUCCESS (Green).")
#                             viz.update_result(current_target_pose, True)
#                             valid_input = True
#                         elif key == 'n':
#                             print(">>> Marked as FAILURE (Red).")
#                             viz.update_result(current_target_pose, False)
#                             valid_input = True
#                         time.sleep(0.01)
                
#                 current_target_pose = None # 重置目标点

#             # =========================================================
#             # 物理复位 (在评估之后执行)
#             # =========================================================
#             print(">>> Performing physical reset...")
#             if aborted:
#                 robot.open_gripper()
#                 time.sleep(0.5)
#             else:
#                 # 如果是自动结束，通常意味着抓到了，闭合夹爪回
#                 robot.close_gripper()
#                 time.sleep(1.0)
            
#             robot.move_home_scripted()

#     except KeyboardInterrupt:
#         print("\nStopping Program...")
#     finally:
#         robot.close()

# if __name__ == "__main__":
#     main()
import time
import cv2
import numpy as np
import sys
import os
import termios
import select
import random
import threading  # 引入线程库
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
# 1. 配置区域 (保持不变)
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
EXECUTE_STEPS = 1 
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
# 改进：后台键盘监听线程 (无阻塞)
# -----------------------------------------------------------------------------
class KeyboardThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.last_key = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        # 设置终端为非规范模式 (无回显，无缓冲)
        fd = sys.stdin.fileno()
        old_term = termios.tcgetattr(fd)
        new_term = termios.tcgetattr(fd)
        new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(fd, termios.TCSANOW, new_term)

        try:
            while self.running:
                # 使用 select 监听，避免死循环占用 CPU
                dr, dw, de = select.select([sys.stdin], [], [], 0.1)
                if dr:
                    key = sys.stdin.read(1)
                    with self.lock:
                        self.last_key = key
        finally:
            # 恢复终端设置
            termios.tcsetattr(fd, termios.TCSANOW, old_term)

    def get_last_key(self):
        with self.lock:
            k = self.last_key
            self.last_key = None # 读取后清空
            return k

    def stop(self):
        self.running = False

# -----------------------------------------------------------------------------
# 2. 可视化 & 采样工具 (保持不变)
# -----------------------------------------------------------------------------
class TaskVisualizer:
    def __init__(self, save_dir, boundary_points, home_pos):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_dir / "performance_map_exp10_58000_0105rainingDay.png"
        self.boundary = boundary_points
        self.home_pos = home_pos
        pad = 50
        min_x, max_x = np.min(boundary_points[:, 0]), np.max(boundary_points[:, 0])
        min_y, max_y = np.min(boundary_points[:, 1]), np.max(boundary_points[:, 1])
        self.scale = 1.5 
        self.offset_x = -min_x * self.scale + pad
        self.offset_y = -min_y * self.scale + pad
        width = int((max_x - min_x) * self.scale + pad * 2)
        height = int((max_y - min_y) * self.scale + pad * 2)
        self.img_h, self.img_w = height, width
        self.canvas = self._load_or_create()

    def _to_pixel(self, x, y):
        u = int(x * self.scale + self.offset_x)
        v = int(y * self.scale + self.offset_y)
        return (u, v)

    def _load_or_create(self):
        if self.save_path.exists():
            img = cv2.imread(str(self.save_path))
        else:
            img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255
            pts = []
            for p in self.boundary: pts.append(self._to_pixel(p[0], p[1]))
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)
            for p in pts: cv2.circle(img, (p[0][0], p[0][1]), 6, (0, 0, 0), -1)
            hu, hv = self._to_pixel(self.home_pos[0], self.home_pos[1])
            cv2.circle(img, (hu, hv), 8, (255, 0, 0), -1)
            cv2.imwrite(str(self.save_path), img)
        return img if img is not None else np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 255

    def update_result(self, pose, success):
        x, y, yaw = pose[0], pose[1], pose[5]
        start_pt = self._to_pixel(x, y)
        color = (0, 255, 0) if success else (0, 0, 255)
        arrow_len_px = 25
        end_x_px = start_pt[0] + arrow_len_px * np.cos(yaw)
        end_y_px = start_pt[1] + arrow_len_px * np.sin(yaw)
        cv2.circle(self.canvas, start_pt, 4, color, -1) 
        cv2.arrowedLine(self.canvas, start_pt, (int(end_x_px), int(end_y_px)), color, thickness=2, tipLength=0.3)
        cv2.imwrite(str(self.save_path), self.canvas)

class TaskSampler:
    def __init__(self):
        self.hull_2d = ConvexHull(BOUNDARY_POINTS_2D)
        self.boundary_points = BOUNDARY_POINTS_2D
        self.min_x = np.min(BOUNDARY_POINTS_2D[:, 0])
        self.max_x = np.max(BOUNDARY_POINTS_2D[:, 0])
        self.min_y = np.min(BOUNDARY_POINTS_2D[:, 1])
        self.max_y = np.max(BOUNDARY_POINTS_2D[:, 1])
        self.grid_rows, self.grid_cols = 4, 4
        self.grid_indices = []
        self._refill_grid_indices()
        
        ccw_indices = self.hull_2d.vertices
        self.ccw_vertices = BOUNDARY_POINTS_2D[ccw_indices]
        self.path_points_2d = self._generate_boundary_path(self.ccw_vertices, 20.0)
        self.total_path_points = len(self.path_points_2d)
        
    def _refill_grid_indices(self):
        self.grid_indices = [(r, c) for r in range(self.grid_rows) for c in range(self.grid_cols)]
        random.shuffle(self.grid_indices)

    def _generate_boundary_path(self, vertices, step_size):
        path = []
        num_v = len(vertices)
        for i in range(num_v):
            p_curr, p_next = vertices[i], vertices[(i + 1) % num_v]
            vec = p_next - p_curr
            dist = np.linalg.norm(vec)
            steps = int(max(1, dist / step_size))
            unit_vec = vec / dist
            for s in range(steps): path.append(p_curr + unit_vec * (s * step_size))
        return np.array(path)

    def is_inside(self, x, y):
        return all(np.dot(eq, np.array([x, y, 1])) <= 1e-6 for eq in self.hull_2d.equations)

    def get_random_grid_target(self):
        for _ in range(32):
            if not self.grid_indices: self._refill_grid_indices()
            r, c = self.grid_indices.pop()
            step_x = (self.max_x - self.min_x) / self.grid_cols
            step_y = (self.max_y - self.min_y) / self.grid_rows
            min_x, max_x = self.min_x + c * step_x, self.min_x + (c+1) * step_x
            min_y, max_y = self.min_y + r * step_y, self.min_y + (r+1) * step_y
            for _ in range(10):
                tx, ty = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
                if self.is_inside(tx, ty):
                    return [tx, ty, FIXED_Z, FIXED_ROLL, FIXED_PITCH, BASE_YAW + random.uniform(*YAW_RANDOM_RANGE)]
        c = np.mean(self.boundary_points, axis=0)
        return [c[0], c[1], FIXED_Z, FIXED_ROLL, FIXED_PITCH, BASE_YAW]

    def get_boundary_target(self):
        pt = self.path_points_2d[random.randint(0, self.total_path_points - 1)]
        return [pt[0], pt[1], FIXED_Z, FIXED_ROLL, FIXED_PITCH, BASE_YAW + random.uniform(*YAW_RANDOM_RANGE)]

# -----------------------------------------------------------------------------
# 3. 硬件封装类 (略微优化)
# -----------------------------------------------------------------------------
class XArmHardware:
    def __init__(self, ip, camera_indices):
        print(f"Connecting to xArm at {ip}...")
        self.arm = XArmAPI(ip)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(enable=True)
        self.arm.set_tgpio_modbus_baudrate(baud=115200)
        self.caps = {}
        for name, idx in camera_indices.items():
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened(): self.caps[name] = cap
            else: print(f"[Warn] Failed to open camera {name}")
        self.current_gripper_state = 0.0
        self.arm.set_mode(0); self.arm.set_state(0)
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
        obs["state"] = np.append(state_vec, self.current_gripper_state)
        return obs

    def move_to_pose_scripted(self, pose, speed=100, wait=True):
        self.arm.set_mode(0); self.arm.set_state(0)
        self.arm.set_position(x=pose[0], y=pose[1], z=pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5], speed=speed, wait=wait, is_radian=True)

    def move_home_scripted(self):
        self.move_to_pose_scripted(HOME_POS, speed=150)

    def run_setup_sequence(self, target_pose):
        pose_A_up = list(POS_A); pose_A_up[2] += 100
        target_up = list(target_pose); target_up[2] += 100
        try:
            self.move_to_pose_scripted(pose_A_up, speed=300)
            self.move_to_pose_scripted(POS_A, speed=150)
            self.close_gripper(); time.sleep(1.5)
            self.move_to_pose_scripted(pose_A_up, speed=300)
            self.move_home_scripted()
            self.move_to_pose_scripted(target_up, speed=300)
            self.move_to_pose_scripted(target_pose, speed=150)
            self.open_gripper(); time.sleep(1.5)
            self.move_to_pose_scripted(target_up, speed=300)
            self.move_home_scripted()
        except Exception: self.move_home_scripted()

    def move_to_start(self, target_action_rad):
        safe_joints = [np.clip(a, l, h) for a, (l, h) in zip(target_action_rad[:6], JOINT_LIMITS)]
        self.arm.set_mode(0); self.arm.set_state(0)
        self.arm.set_servo_angle(angle=safe_joints, speed=0.35, is_radian=True, wait=True)
        self.arm.set_mode(1); self.arm.set_state(0)
        time.sleep(0.5)

    def execute_action(self, action_rad):
        # 简化版执行，去除不必要的打印和逻辑，保持高效
        safe_joints = [np.clip(a, l, h) for a, (l, h) in zip(action_rad[:6], JOINT_LIMITS)]
        
        # 获取当前角度 (通信耗时)
        code, current_joints = self.arm.get_servo_angle(is_radian=True)
        if code != 0: return

        curr_j = np.array(current_joints[:6])
        targ_j = np.array(safe_joints[:6])
        
        duration = (1.0 / CONTROL_FREQ) * SLOW_DOWN_FACTOR
        steps = int(duration * INTERPOLATION_FREQ)
        if steps < 1: steps = 1
        
        # 插值循环 (纯计算+发送)
        for i in range(1, steps + 1):
            # 不要在插值循环内做任何 IO (如键盘检查)
            alpha = i / steps
            interp = curr_j + (targ_j - curr_j) * alpha
            full = np.append(interp, 0.0)
            self.arm.set_servo_angle_j(angles=full, is_radian=True)
            time.sleep(1.0 / INTERPOLATION_FREQ)

        target_gripper = action_rad[6]
        if target_gripper > 0.8: self.close_gripper()
        elif target_gripper < 0.2: self.open_gripper()

    def close(self):
        self.arm.set_mode(0); self.arm.set_state(0)
        for cap in self.caps.values(): cap.release()
        self.arm.disconnect()

# -----------------------------------------------------------------------------
# 4. 主程序 (重构)
# -----------------------------------------------------------------------------
def main():
    print(f"Loading Config & Model...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    print("Model Loaded.")

    robot = XArmHardware(ROBOT_IP, CAMERAS)
    sampler = TaskSampler()
    viz = TaskVisualizer(VIS_SAVE_DIR, BOUNDARY_POINTS_2D, HOME_POS)
    
    # === 启动键盘监听线程 ===
    kb_thread = KeyboardThread()
    kb_thread.start()
    
    prompt = "pick up the industrial components"
    current_target_pose = None
    
    try:
        episode_count = 0
        while True:
            episode_count += 1
            print(f"\n" + "="*50)
            print(f"Episode {episode_count}")
            print("Commands: [r]andom, [b]oundary, [ENTER] start, [o] reset")
            print("="*50)

            # --- 等待用户指令 (Setting Mode) ---
            mode = None 
            while mode is None:
                key = kb_thread.get_last_key()
                if key:
                    if key == '\n' or key == '\r':
                        mode = 'inference'
                    elif key == 'r':
                        print(">>> Random Grid Setup...")
                        current_target_pose = sampler.get_random_grid_target()
                        robot.run_setup_sequence(current_target_pose)
                        print(">>> Ready. Press ENTER.")
                    elif key == 'b': 
                        print(">>> Boundary Setup...")
                        current_target_pose = sampler.get_boundary_target()
                        robot.run_setup_sequence(current_target_pose)
                        print(">>> Ready. Press ENTER.")
                    elif key == 'o': 
                        robot.open_gripper(); robot.move_home_scripted()
                        print(">>> Reset Done.")
                time.sleep(0.05) # 设置阶段可以睡眠长一点

            # --- 预对齐 ---
            print(">>> Pre-inferencing...")
            raw_obs = robot.get_observation()
            example = {
                "cam_left_wrist": raw_obs["cam_left_wrist"],
                "cam_right_wrist": raw_obs["cam_right_wrist"],
                "state": raw_obs["state"], "prompt": prompt
            }
            result = policy.infer(example)
            robot.move_to_start(np.array(result["actions"])[0])

            # === AI Control Loop (Real-time) ===
            print(">>> AI Loop Started (Press 'o' to STOP)...")
            aborted = False

            while True:
                # 1. 极速检查退出标志 (内存读取，无IO阻塞)
                if kb_thread.get_last_key() == 'o':
                    print("\n>>> EMERGENCY STOP (User).")
                    aborted = True
                    break

                # 2. 观测
                raw_obs = robot.get_observation()
                example = {
                    "cam_left_wrist": raw_obs["cam_left_wrist"],
                    "cam_right_wrist": raw_obs["cam_right_wrist"],
                    "state": raw_obs["state"], "prompt": prompt
                }

                # 3. 推理
                result = policy.infer(example)
                action_chunk = np.array(result["actions"]) 

                # 4. 抓取检测
                if np.any(action_chunk[:1, 6] > 0.8):
                    print(f"\n>>> Grasp detected (Auto Stop).")
                    break

                # 5. 执行 (Chunk Loop)
                steps_to_run = min(EXECUTE_STEPS, len(action_chunk))
                for i in range(steps_to_run):
                    st = time.time()
                    # 再次检查停止 (可选，增加安全性，开销极小)
                    if kb_thread.get_last_key() == 'o': aborted = True; break
                    
                    robot.execute_action(action_chunk[i]) # 内部严格控制了时间
                    
                    # 频率补偿 (只补不足的时间，不额外睡眠)
                    el = time.time() - st
                    remaining = (1.0/CONTROL_FREQ) - el
                    if remaining > 0: time.sleep(remaining)
                
                if aborted: break

            # === 结果评估 ===
            if current_target_pose is not None:
                print("\n>>> Evaluation: [y] Success / [n] Failure")
                valid = False
                while not valid:
                    k = kb_thread.get_last_key()
                    if k == 'y':
                        viz.update_result(current_target_pose, True)
                        valid = True
                    elif k == 'n':
                        viz.update_result(current_target_pose, False)
                        valid = True
                    time.sleep(0.05)
                current_target_pose = None

            # === 复位 ===
            if aborted: robot.open_gripper()
            else: robot.close_gripper()
            time.sleep(1.5)
            robot.move_home_scripted()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        kb_thread.stop() # 停止监听线程
        robot.close()

if __name__ == "__main__":
    main()