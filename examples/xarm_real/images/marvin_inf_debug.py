#!/usr/bin/env python3
"""
Marvin 推理调试脚本 - 最小化版本
只执行模型推理，打印详细的action和IK信息
"""
import os
import sys
import time
import numpy as np
import cv2

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from SDK_PYTHON.fx_kine import Marvin_Kine, FX_InvKineSolvePara
from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS

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

# ==================== 配置 ====================
ROBOT_IP = "10.10.13.12"
ARM_NAME = 'A'
ARM_TYPE = 0
CONFIG_FILE = os.path.join(root_dir, 'SDK_PYTHON', 'ccs_m6_40.MvKDCfg')

# 模型配置
MODEL_NAME = "pi05_xarm_1212_night"
CHECKPOINT = 24000
TASK_PROMOT = "pick up the object"

# 相机配置
CAM_LEFT_WRIST = "/dev/video0"
CAM_RIGHT_WRIST = "/dev/video2"

# 控制参数
CONTROL_FREQ = 10
EXECUTE_STEPS = 2
VEL_RATIO_SERVO = 5
ACC_RATIO = 10

# ==================== 机器人硬件类 ====================
class MarvinHardwareDebug:
    def __init__(self, ip, config_file):
        print(f"[Init] 连接MARVIN机器人 {ip}...")
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
        
        self.arm.clear_set()
        self.arm.set_state(arm=ARM_NAME, state=1)
        self.arm.set_vel_acc(arm=ARM_NAME, velRatio=VEL_RATIO_SERVO, AccRatio=ACC_RATIO)
        self.arm.send_cmd()
        time.sleep(0.5)
        print("[Init] ✅ 机器人连接成功")

        # 初始化相机
        self.caps = {}
        try:
            cap_left = cv2.VideoCapture(CAM_LEFT_WRIST)
            if cap_left.isOpened():
                cap_left.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.caps['cam_left_wrist'] = cap_left
                print(f"[Init] ✅ 左腕相机已连接: {CAM_LEFT_WRIST}")
        except Exception as e:
            print(f"[Init] ⚠️ 左腕相机连接失败: {e}")

        try:
            cap_right = cv2.VideoCapture(CAM_RIGHT_WRIST)
            if cap_right.isOpened():
                cap_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.caps['cam_right_wrist'] = cap_right
                print(f"[Init] ✅ 右腕相机已连接: {CAM_RIGHT_WRIST}")
        except Exception as e:
            print(f"[Init] ⚠️ 右腕相机连接失败: {e}")

        self.current_gripper_state = 0.0

    def get_current_cartesian(self):
        """获取当前笛卡尔位姿 (米, 弧度)"""
        sub_data = self.arm.subscribe(self.dcss)
        arm_idx = 0 if ARM_NAME == 'A' else 1
        fb_joint_pos = sub_data['outputs'][arm_idx]['fb_joint_pos']
        
        # 正向运动学
        fk_result = self.kine.fk(joints=fb_joint_pos)
        if not fk_result:
            return None
        
        pose_mat = fk_result.m_Output_TcpPose
        xyzabc = self.kine.pose_mat_to_xyzabc(pose_mat)
        if xyzabc is None:
            return None
        
        # 转换为米和弧度
        pose = np.array([
            xyzabc[0] / 1000.0,  # mm -> m
            xyzabc[1] / 1000.0,
            xyzabc[2] / 1000.0,
            np.radians(xyzabc[3]),  # deg -> rad
            np.radians(xyzabc[4]),
            np.radians(xyzabc[5])
        ])
        return pose

    def get_observation(self):
        """获取观测图像"""
        obs = {}
        for cam_name, cap in self.caps.items():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                obs[cam_name] = frame_rgb
            else:
                # 创建黑图
                if 'cam_right_wrist' in obs:
                    h, w = obs['cam_right_wrist'].shape[:2]
                    obs[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    obs[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 确保至少有两个相机
        if 'cam_left_wrist' not in obs:
            if 'cam_right_wrist' in obs:
                h, w = obs['cam_right_wrist'].shape[:2]
                obs['cam_left_wrist'] = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                obs['cam_left_wrist'] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if 'cam_right_wrist' not in obs:
            if 'cam_left_wrist' in obs:
                h, w = obs['cam_left_wrist'].shape[:2]
                obs['cam_right_wrist'] = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                obs['cam_right_wrist'] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        return obs

    def check_ik(self, target_pose_m, current_joints=None):
        """
        检查IK解算
        target_pose_m: [x, y, z, rx, ry, rz] (米, 弧度)
        返回: (success, joints, info_dict)
        """
        # 转换为毫米和度
        target_xyzabc = target_pose_m.copy()
        target_xyzabc[:3] *= 1000.0  # 米转毫米
        target_xyzabc[3:] = np.degrees(target_xyzabc[3:])  # 弧度转度

        # 获取当前关节角
        if current_joints is None:
            sub_data = self.arm.subscribe(self.dcss)
            arm_idx = 0 if ARM_NAME == 'A' else 1
            current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
        
        current_joints = [float(j) for j in current_joints]

        # IK解算
        target_pose_mat = self.kine.xyzabc_to_mat4x4(xyzabc=target_xyzabc)
        ik_para = FX_InvKineSolvePara()
        mat16 = self.kine.mat4x4_to_mat1x16(target_pose_mat)
        ik_para.set_input_ik_target_tcp(mat16)
        ik_para.set_input_ik_ref_joint(current_joints)
        ik_para.set_input_ik_zsp_type(0)

        ik_result = self.kine.ik(structure_data=ik_para)

        info = {
            'target_xyzabc_mm': target_xyzabc[:3],
            'target_rpy_deg': target_xyzabc[3:],
            'current_joints': current_joints[:7],
        }

        if ik_result and not ik_result.m_Output_IsOutRange and not ik_result.m_Output_IsJntExd:
            target_joints = ik_result.m_Output_RetJoint.to_list()
            info['target_joints'] = target_joints[:7]
            info['joint_diff'] = np.array(target_joints[:7]) - np.array(current_joints[:7])
            info['max_joint_diff'] = np.max(np.abs(info['joint_diff']))
            return True, target_joints, info
        else:
            info['ik_error'] = "OutRange" if ik_result and ik_result.m_Output_IsOutRange else "JntExd" if ik_result and ik_result.m_Output_IsJntExd else "IK Failed"
            return False, None, info

    def execute_action(self, action_delta, start_pose_abs, current_pose_abs):
        """
        执行动作（简化版，只打印信息，不实际执行）
        action_delta: 模型输出的相对action (相对于start_pose_abs)
        """
        # 计算预测目标位置
        pred_target_abs = start_pose_abs[:6] + action_delta[:6]
        
        # 计算实际需要执行的delta
        real_delta = pred_target_abs - current_pose_abs
        
        # 检查IK
        success, joints, info = self.check_ik(pred_target_abs, None)
        
        print(f"\n{'='*80}")
        print(f"[Action Debug]")
        print(f"  Model output (relative to start):")
        print(f"    XYZ: {action_delta[:3]*1000:.2f} mm")
        print(f"    RPY: {np.degrees(action_delta[3:6]):.1f} deg")
        print(f"  Start pose (abs):")
        print(f"    XYZ: {start_pose_abs[:3]*1000:.2f} mm")
        print(f"    RPY: {np.degrees(start_pose_abs[3:6]):.1f} deg")
        print(f"  Current pose (abs):")
        print(f"    XYZ: {current_pose_abs[:3]*1000:.2f} mm")
        print(f"    RPY: {np.degrees(current_pose_abs[3:6]):.1f} deg")
        print(f"  Predicted target (abs):")
        print(f"    XYZ: {pred_target_abs[:3]*1000:.2f} mm")
        print(f"    RPY: {np.degrees(pred_target_abs[3:6]):.1f} deg")
        print(f"  Real delta (current -> target):")
        print(f"    XYZ: {real_delta[:3]*1000:.2f} mm")
        print(f"    RPY: {np.degrees(real_delta[3:6]):.1f} deg")
        print(f"  IK Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        if success:
            print(f"    Target joints: {[f'{j:.2f}' for j in info['target_joints']]}")
            print(f"    Current joints: {[f'{j:.2f}' for j in info['current_joints']]}")
            print(f"    Joint diff: {[f'{j:.2f}' for j in info['joint_diff']]}")
            print(f"    Max joint diff: {info['max_joint_diff']:.2f} deg")
        else:
            print(f"    Error: {info.get('ik_error', 'Unknown')}")
        print(f"{'='*80}\n")
        
        return success

    def close(self):
        if hasattr(self, 'arm'):
            self.arm.set_state(arm=ARM_NAME, state=0)
            self.arm.send_cmd()
            self.arm.release_robot()
        for cap in self.caps.values():
            cap.release()

# ==================== 主函数 ====================
def main():
    print("="*80)
    print("Marvin 推理调试脚本")
    print("="*80)
    
    # 1. 初始化机器人
    robot = None
    try:
        robot = MarvinHardwareDebug(ROBOT_IP, CONFIG_FILE)
    except Exception as e:
        print(f"[Error] 初始化失败: {e}")
        return

    # 2. 加载模型
    print(f"\n[Model] 加载模型: {MODEL_NAME} @ checkpoint {CHECKPOINT}")
    try:
        client = OpenPIClient()
        policy = client.load_policy(MODEL_NAME, checkpoint=CHECKPOINT)
        print("[Model] ✅ 模型加载成功")
    except Exception as e:
        print(f"[Error] 模型加载失败: {e}")
        robot.close()
        return

    # 3. 获取起始位置
    print("\n[Init] 获取起始位置...")
    robot.flush_cameras()
    time.sleep(0.5)
    start_pose_abs = robot.get_current_cartesian()
    if start_pose_abs is None:
        print("[Error] 无法获取起始位置")
        robot.close()
        return
    
    print(f"[Init] 起始位置:")
    print(f"  XYZ: {start_pose_abs[:3]*1000:.2f} mm")
    print(f"  RPY: {np.degrees(start_pose_abs[3:6]):.1f} deg")

    # 4. 推理循环
    print("\n" + "="*80)
    print("开始推理循环...")
    print("按 Ctrl+C 退出")
    print("="*80 + "\n")

    step_count = 0
    try:
        while True:
            step_count += 1
            print(f"\n>>> Step {step_count} <<<")
            
            # 4.1 获取观测
            raw_obs = robot.get_observation()
            current_pose_abs = robot.get_current_cartesian()
            
            # 4.2 构造输入状态（相对位姿）
            rel_pose = current_pose_abs - start_pose_abs
            # 归一化角度
            rel_pose[5] = (rel_pose[5] + np.pi) % (2 * np.pi) - np.pi
            
            input_state = np.append(rel_pose, robot.current_gripper_state)
            print(f"[State] Rel_pose XYZ: {rel_pose[:3]*1000:.2f} mm, RPY: {np.degrees(rel_pose[3:6]):.1f} deg")
            
            # 4.3 模型推理
            result = policy.infer({
                "cam_left_wrist": raw_obs["cam_left_wrist"],
                "cam_right_wrist": raw_obs["cam_right_wrist"],
                "state": input_state,
                "prompt": TASK_PROMOT
            })
            
            action_chunk = np.array(result["actions"])
            print(f"[Model] Action chunk shape: {action_chunk.shape}")
            
            # 4.4 处理每个action
            steps_to_run = min(EXECUTE_STEPS, len(action_chunk))
            for i in range(steps_to_run):
                raw_action = action_chunk[i]
                print(f"\n--- Action {i+1}/{steps_to_run} in chunk ---")
                
                # 检查IK并打印详细信息
                success = robot.execute_action(raw_action, start_pose_abs, current_pose_abs)
                
                if not success:
                    print("[Warning] IK失败，跳过此action")
                    break
            
            # 等待
            time.sleep(1.0 / CONTROL_FREQ)
            
    except KeyboardInterrupt:
        print("\n\n[Exit] 用户中断")
    except Exception as e:
        print(f"\n[Error] 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.close()
        print("[Exit] 已关闭连接")

if __name__ == "__main__":
    main()
