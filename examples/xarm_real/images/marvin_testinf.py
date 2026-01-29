# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import time
# import numpy as np
# import sys
# import os
# import json
# import cv2

# # ================= 配置路径 =================
# # OpenPI 依赖
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# openpi_client_path = os.path.join(current_dir, "../../../packages/openpi-client/src")
# sys.path.append("/home/openpi/src")
# sys.path.append(os.path.abspath(openpi_client_path))
# sys.path.insert(0, parent_dir)

# from openpi.training import config as _config
# from openpi.policies import policy_config
# from openpi_client import image_tools

# # Marvin SDK
# root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# if root_dir not in sys.path:
#     sys.path.insert(0, root_dir)
# from SDK_PYTHON.fx_kine import Marvin_Kine, FX_InvKineSolvePara
# from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS

# # ================= 基础配置 =================
# ROBOT_IP = "10.10.13.12"
# CONFIG_NAME = "pi05_xarm_1212_night"
# CHECKPOINT_DIR = "/home/openpi/checkpoints/exp1/1000"
# CONFIG_FILE = os.path.join(root_dir, 'SDK_PYTHON', 'ccs_m6_40.MvKDCfg')
# TASK_PROMPT = "pick up the object"
# CAMERAS = {"cam_right_wrist": 2}

# # ================= 硬件类 =================
# class MarvinCheck:
#     def __init__(self, ip, config_file):
#         print(f"Connecting to MARVIN at {ip}...")
#         self.dcss = DCSS()
#         self.arm = Marvin_Robot()
#         self.kine = Marvin_Kine()
#         # self.kine.log_switch(0) # 关闭SDK内部日志，保持清爽
        
#         self.arm.connect(ip)
#         self.kine.load_config(arm_type=0, config_path=config_file)
#         # 初始化运动学参数 (这里假设配置文件加载成功，简化代码)
#         ini_result = self.kine.load_config(arm_type=0, config_path=config_file)
#         if ini_result:
#             self.kine.initial_kine(ini_result['TYPE'][0], ini_result['DH'][0], ini_result['PNVA'][0], ini_result['BD'][0])
        
#         # 相机
#         self.caps = {}
#         for name, idx in CAMERAS.items():
#             self.caps[name] = cv2.VideoCapture(idx)
        
#         time.sleep(1.0)

#     def get_obs_and_pose(self):
#         # 1. 图像
#         obs = {}
#         for name, cap in self.caps.items():
#             ret, frame = cap.read()
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else np.zeros((480,640,3), dtype=np.uint8)
#             obs[name] = image_tools.convert_to_uint8(frame)
        
#         if "cam_left_wrist" not in obs: obs["cam_left_wrist"] = np.zeros((480,640,3), dtype=np.uint8)
#         obs["cam_high"] = np.zeros((224,224,3), dtype=np.uint8)

#         # 2. 关节和位姿
#         sub_data = self.arm.subscribe(self.dcss)
#         joints_deg = sub_data['outputs'][0]['fb_joint_pos'] # 7轴角度
        
#         # 计算 FK 获取当前 XYZ(mm) RPY(deg)
#         pose_mat = self.kine.fk(joints=joints_deg)
#         curr_pose_mm_deg = np.array(self.kine.mat4x4_to_xyzabc(pose_mat=pose_mat)) # [mm, mm, mm, deg, deg, deg]
        
#         # 转换成 [米, 弧度] 供模型使用
#         curr_pose_m_rad = curr_pose_mm_deg.copy()
#         curr_pose_m_rad[:3] /= 1000.0
#         curr_pose_m_rad[3:] = np.radians(curr_pose_m_rad[3:])
        
#         # State 输入 (取前6个关节弧度)
#         input_joints = [np.radians(j) for j in joints_deg[:6]]
#         obs["state"] = np.append(input_joints, 0.0) # 0.0 gripper
        
#         return obs, curr_pose_m_rad, joints_deg

#     def run_ik_check(self, target_pose_mm_deg, current_joints_deg):
#         """
#         运行 IK 并计算关节差值
#         target_pose_mm_deg: [x, y, z, r, p, y] (毫米, 度)
#         """
#         pose_mat = self.kine.xyzabc_to_mat4x4(xyzabc=target_pose_mm_deg)
#         ik_para = FX_InvKineSolvePara()
#         mat16 = self.kine.mat4x4_to_mat1x16(pose_mat)
#         ik_para.set_input_ik_target_tcp(mat16)
#         ik_para.set_input_ik_ref_joint(current_joints_deg)
#         ik_para.set_input_ik_zsp_type(0)
        
#         ik_result = self.kine.ik(structure_data=ik_para)
        
#         if not ik_result or ik_result.m_Output_IsOutRange or ik_result.m_Output_IsJntExd:
#             return None, "FAIL"
        
#         target_joints = ik_result.m_Output_RetJoint.to_list()
        
#         # 计算最大关节跳变
#         diffs = [t - c for t, c in zip(target_joints, current_joints_deg)]
#         max_diff = max(np.abs(diffs))
        
#         return target_joints, f"SUCCESS (Max Diff: {max_diff:.2f}°)"

#     def close(self):
#         self.arm.release_robot()

# # ================= 主程序 =================
# def main():
#     print(">>> Loading Model...")
#     config = _config.get_config(CONFIG_NAME)
#     policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
#     robot = MarvinCheck(ROBOT_IP, CONFIG_FILE)
    
#     try:
#         print("\n>>> 获取当前状态...")
#         obs, start_pose_m_rad, start_joints_deg = robot.get_obs_and_pose()
        
#         # 打印当前状态
#         print(f"Start Pose (Meters/Rad): {start_pose_m_rad}")
#         print(f"Start Joints (Deg):      {start_joints_deg}")
        
#         print("\n>>> 正在推理...")
#         # 构造输入 (假设相对第一帧)
#         input_state = obs["state"] # 实际上这里应该减去 start_pose，简单起见先直接传
#         # 为了更准确模拟你的代码逻辑：
#         # rel_pose = curr - start = 0
#         obs["state"][:6] = 0.0 
        
#         result = policy.infer({
#             "cam_left_wrist": obs["cam_left_wrist"],
#             "cam_right_wrist": obs["cam_right_wrist"],
#             "state": obs["state"], 
#             "prompt": TASK_PROMPT
#         })
        
#         action_chunk = np.array(result["actions"])
#         print(f">>> 推理完成，分析前 5 步 Action...")
        
#         for i in range(5):
#             print(f"\n--- Step {i} ---")
#             raw_action = action_chunk[i] # [dx, dy, dz, dr, dp, dy, g] (米, 弧度)
#             print(f"1. Model Output (Raw Action): {raw_action[:6]}")
            
#             # --- 模拟你之前的错误逻辑 ---
#             wrong_target_abs = start_pose_m_rad[:6] + raw_action[:6]
#             wrong_target_mm = wrong_target_abs * 1000.0 
#             print(f"2. [错误] 乘以1000后的输入值: {wrong_target_mm}")
#             print(f"   注意旋转项: {wrong_target_mm[3:]} -> 这被当作了'度'传入IK！")
            
#             # --- 正确的转换逻辑 ---
#             # 1. 位置: 米 -> 毫米 (*1000)
#             target_pos_mm = (start_pose_m_rad[:3] + raw_action[:3]) * 1000.0
#             # 2. 旋转: 弧度 -> 弧度 (相加) -> 度 (转换)
#             target_rot_rad = start_pose_m_rad[3:] + raw_action[3:6]
#             target_rot_deg = np.degrees(target_rot_rad)
            
#             correct_target_mm_deg = np.concatenate([target_pos_mm, target_rot_deg])
#             print(f"3. [正确] 单位转换后的目标值: {correct_target_mm_deg}")
            
#             # --- 验证 IK ---
#             joints, status = robot.run_ik_check(correct_target_mm_deg, start_joints_deg)
#             print(f"4. IK 解算结果 (基于正确值): {status}")
#             if joints:
#                 print(f"   Target Joints: {joints}")
#                 # 检查差值
#                 diff = np.array(joints) - np.array(start_joints_deg)
#                 print(f"   Diff from Start: {diff}")

#     finally:
#         robot.close()

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import sys
import os
import json
import cv2
import torch

# ================= 配置路径 =================
# OpenPI 依赖 (保持你的路径设置)
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
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from SDK_PYTHON.fx_kine import Marvin_Kine, FX_InvKineSolvePara
from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS

# ================= 基础配置 =================
ROBOT_IP = "10.10.13.12"
CONFIG_NAME = "pi05_xarm_1212_night"
CHECKPOINT_DIR = "/home/openpi/checkpoints/exp1/1000"
CONFIG_FILE = os.path.join(root_dir, 'SDK_PYTHON', 'ccs_m6_40.MvKDCfg')
TASK_PROMPT = "pick up the object"
CAMERAS = {"cam_right_wrist": 2}

# ================= 硬件辅助类 =================
class MarvinInferenceCheck:
    def __init__(self, ip, config_file):
        print(f"Connecting to MARVIN at {ip}...")
        self.dcss = DCSS()
        self.arm = Marvin_Robot()
        self.kine = Marvin_Kine()
        # 关闭 SDK 内部打印，便于观察我们自己的输出
        self.kine.log_switch(0) 
        
        self.arm.connect(ip)
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件未找到: {config_file}")
            
        ini_result = self.kine.load_config(arm_type=0, config_path=config_file)
        if ini_result:
            self.kine.initial_kine(ini_result['TYPE'][0], ini_result['DH'][0], ini_result['PNVA'][0], ini_result['BD'][0])
        
        self.caps = {}
        for name, idx in CAMERAS.items():
            self.caps[name] = cv2.VideoCapture(idx)
            self.caps[name].set(3, 640)
            self.caps[name].set(4, 480)
        
        time.sleep(1.0)

    def get_current_state(self):
        """
        获取当前状态
        返回: 
        1. obs (包含图像和 state)
        2. start_pose_m_deg: 当前绝对位姿 [米, 度] (用于加上模型输出的相对值)
        3. current_joints_deg: 当前关节角度 [度] (用于 IK 参考和对比)
        """
        # --- 1. 获取图像 ---
        obs = {}
        for name, cap in self.caps.items():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                obs[name] = image_tools.convert_to_uint8(frame)
            else:
                obs[name] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 补全缺失图像
        if "cam_left_wrist" not in obs: obs["cam_left_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
        obs["cam_high"] = np.zeros((224, 224, 3), dtype=np.uint8)

        # --- 2. 获取关节和位姿 ---
        sub_data = self.arm.subscribe(self.dcss)
        joints_deg = sub_data['outputs'][0]['fb_joint_pos'] # 原始 7 轴角度
        
        # FK 正解算: 关节(度) -> 笛卡尔(毫米, 度)
        pose_mat = self.kine.fk(joints=joints_deg)
        curr_pose_mm_deg = np.array(self.kine.mat4x4_to_xyzabc(pose_mat=pose_mat))
        
        # 单位转换: [毫米, 度] -> [米, 度]
        # 这是为了作为 Base Pose，因为模型输出的是基于 [米, 度] 的增量
        curr_pose_m_deg = curr_pose_mm_deg.copy()
        curr_pose_m_deg[:3] /= 1000.0 
        
        # --- 3. 构造模型输入 State ---
        # 你的训练逻辑: State = Current_Pose - Start_Pose
        # 在推理的第一帧，Current == Start，所以 State 为全 0
        # 这里的 state 是相对笛卡尔位姿 [rel_x, rel_y, rel_z, rel_r, rel_p, rel_y, gripper]
        input_state = np.zeros(7, dtype=np.float32)
        # 夹爪状态 (假设开启为1.0)
        input_state[6] = 0.0 
        
        obs["state"] = input_state
        
        return obs, curr_pose_m_deg, joints_deg

    def calculate_ik(self, target_pose_mm_deg, ref_joints):
        """
        逆解算: 目标位姿(毫米, 度) -> 目标关节角
        """
        pose_mat = self.kine.xyzabc_to_mat4x4(xyzabc=target_pose_mm_deg)
        ik_para = FX_InvKineSolvePara()
        mat16 = self.kine.mat4x4_to_mat1x16(pose_mat)
        
        ik_para.set_input_ik_target_tcp(mat16)
        ik_para.set_input_ik_ref_joint(ref_joints)
        ik_para.set_input_ik_zsp_type(0) # 0: 自动选择最优解
        
        ik_result = self.kine.ik(structure_data=ik_para)
        
        if ik_result:
            # 只要有返回值，即使标志位是 Fail，我们也打印出来看看是否合理
            joints = ik_result.m_Output_RetJoint.to_list()
            is_valid = not (ik_result.m_Output_IsOutRange or ik_result.m_Output_IsJntExd)
            return joints, is_valid
        return None, False

    def close(self):
        self.arm.release_robot()
        for cap in self.caps.values(): cap.release()

# ================= 主程序 =================
def main():
    print(">>> Loading Model...")
    config = _config.get_config(CONFIG_NAME)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    robot = MarvinInferenceCheck(ROBOT_IP, CONFIG_FILE)
    
    try:
        print("\n>>> [1] 获取当前状态...")
        obs, start_pose_m_deg, start_joints_deg = robot.get_current_state()
        
        print(f"Start Pose [M, Deg]: {np.round(start_pose_m_deg, 4)}")
        print(f"Start Joints [Deg]:  {np.round(start_joints_deg, 2)}")
        
        print("\n>>> [2] 正在推理...")
        t0 = time.time()
        result = policy.infer({
            "cam_left_wrist": obs["cam_left_wrist"],
            "cam_right_wrist": obs["cam_right_wrist"],
            "state": obs["state"], 
            "prompt": TASK_PROMPT
        })
        print(f"推理耗时: {(time.time()-t0)*1000:.1f} ms")
        
        action_chunk = np.array(result["actions"])
        print(f"Action Chunk Size: {len(action_chunk)}")
        
        print("\n>>> [3] 分析轨迹 (前 10 步)...")
        print(f"{'Step':<4} | {'Model Out (Rel M, Deg)':<30} | {'Target (Abs mm, Deg)':<30} | {'Joint Diff (Max Deg)'}")
        print("-" * 110)
        
        # 遍历前 10 步
        for i in range(min(10, len(action_chunk))):
            raw_action = action_chunk[i] # [dx, dy, dz, dr, dp, dy, g]
            
            # --- 核心计算逻辑 ---
            
            # 1. 计算绝对目标位姿 [米, 度]
            # Target = Start + Relative_Action
            # 注意：这里的加法直接包含 RPY 的加法 (度 + 度)
            target_pose_m_deg = start_pose_m_deg + raw_action[:6]
            
            # 2. 转换为 SDK 单位 [毫米, 度]
            # 只有 XYZ 需要 * 1000，RPY 保持不变 (因为已经是度了)
            target_pose_mm_deg = target_pose_m_deg.copy()
            target_pose_mm_deg[:3] *= 1000.0
            
            # 3. 逆解算 (IK)
            # 使用当前的 start_joints 作为参考解
            target_joints, is_valid = robot.calculate_ik(target_pose_mm_deg, start_joints_deg)
            
            # 4. 打印结果
            model_out_str = f"{np.round(raw_action[:3], 4)}, {np.round(raw_action[3:6], 2)}"
            target_str = f"{np.round(target_pose_mm_deg[:3], 1)}, {np.round(target_pose_mm_deg[3:], 1)}"
            
            ik_status = "IK FAIL"
            if target_joints:
                # 计算关节变化量
                diff = np.array(target_joints) - np.array(start_joints_deg)
                max_diff = np.max(np.abs(diff)) # 最大单轴变化
                
                status_mark = "✅" if is_valid else "⚠️"
                if max_diff > 45.0: status_mark = "❌ (Jump!)"
                
                ik_status = f"{status_mark} Max Diff: {max_diff:.2f}°"
                
                # 如果是第一步，打印详细关节值
                if i == 0:
                    print(f"    [Step 0 Joints]: {np.round(target_joints, 2).tolist()}")
            
            print(f"{i:<4} | {model_out_str:<30} | {target_str:<30} | {ik_status}")

    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.close()

if __name__ == "__main__":
    main()