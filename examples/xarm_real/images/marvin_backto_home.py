#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marvin机械臂回到Home点脚本
使用关节角度控制，将机械臂移动到HOME_JOINTS位置
"""

import time
import sys
import os

# Marvin SDK
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# 将该目录加入 Python 搜索路径
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from SDK_PYTHON.fx_kine import Marvin_Kine
from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS

# -----------------------------------------------------------------------------
# 配置参数
# -----------------------------------------------------------------------------
ROBOT_IP = "10.10.13.12"

# Home点：关节角度（度）
HOME_JOINTS = [83.20, 31.53, -22.34, 58.78, -79.30, 13.48, 88.53]

# Marvin配置
ARM_NAME = 'A'
ARM_TYPE = 0
CONFIG_FILE = os.path.join(root_dir, 'SDK_PYTHON', 'ccs_m6_40.MvKDCfg')

# 运动速度（百分比）
VEL_RATIO_FAST = 30
ACC_RATIO = 10


# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Marvin机械臂回到Home点")
    print("=" * 60)
    print(f"目标关节角度: {HOME_JOINTS}")
    print(f"机器人IP: {ROBOT_IP}")
    print()
    
    dcss = DCSS()
    arm = Marvin_Robot()
    kine = Marvin_Kine()
    kine.log_switch(0)
    
    try:
        # 1. 连接机器人
        print("[1/4] 正在连接机器人...")
        init = arm.connect(ROBOT_IP)
        if init == 0:
            raise ConnectionError("Failed to connect to MARVIN robot")
        print("✓ 连接成功")
        
        time.sleep(0.5)
        
        # 2. 清除错误
        print("[2/4] 清除错误状态...")
        arm.clear_set()
        arm.clear_error('A')
        arm.clear_error('B')
        arm.send_cmd()
        time.sleep(0.5)
        print("✓ 错误已清除")
        
        # 3. 加载运动学配置
        print("[3/4] 加载运动学配置...")
        if not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
        
        ini_result = kine.load_config(arm_type=ARM_TYPE, config_path=CONFIG_FILE)
        if not ini_result:
            raise RuntimeError("Failed to load kinematics config")
        
        kine.initial_kine(
            robot_type=ini_result['TYPE'][0],
            dh=ini_result['DH'][0],
            pnva=ini_result['PNVA'][0],
            j67=ini_result['BD'][0]
        )
        print("✓ 运动学配置已加载")
        
        # 4. 移动到Home点
        print("[4/4] 移动到Home点...")
        arm.clear_set()
        arm.set_state(arm=ARM_NAME, state=1)  # 位置跟随模式
        arm.set_vel_acc(arm=ARM_NAME, velRatio=VEL_RATIO_FAST, AccRatio=ACC_RATIO)
        arm.set_joint_cmd_pose(arm=ARM_NAME, joints=HOME_JOINTS)
        arm.send_cmd()
        print("✓ 移动指令已发送")
        
        # 等待运动完成
        print("\n等待机械臂移动到Home点（约2-3秒）...")
        time.sleep(3.0)
        
        # 验证当前位置
        print("\n验证当前位置...")
        sub_data = arm.subscribe(dcss)
        arm_idx = 0 if ARM_NAME == 'A' else 1
        current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
        
        print(f"当前关节角度: {[f'{j:.2f}' for j in current_joints[:7]]}")
        print(f"目标关节角度: {[f'{j:.2f}' for j in HOME_JOINTS]}")
        
        # 计算误差
        errors = [abs(c - t) for c, t in zip(current_joints[:7], HOME_JOINTS)]
        max_error = max(errors)
        print(f"\n最大角度误差: {max_error:.2f}°")
        
        if max_error < 2.0:
            print("✓ 已成功到达Home点！")
        else:
            print(f"⚠ 警告：角度误差较大 ({max_error:.2f}°)，可能需要检查")
        
        print("\n" + "=" * 60)
        print("完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 清理
        print("\n清理连接...")
        try:
            arm.set_state(arm=ARM_NAME, state=0)
            arm.send_cmd()
            arm.release_robot()
            print("✓ 连接已关闭")
        except:
            pass
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
