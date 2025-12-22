import time
import numpy as np
from xarm.wrapper import XArmAPI

# ================= 配置区域 =================
ROBOT_IP = "192.168.1.232"

# 你提供的目标角度 (PRED(Rad) 列)
# 这是一个弧度列表，代表 J1 到 J6 的目标位置
TARGET_JOINTS_RAD = [
    -0.2264,  # J1
    -0.8792,  # J2
    -1.5597,  # J3
    -3.2276,  # J4
    0.6412,   # J5
    1.8786    # J6
]

# 夹爪动作：-0.0042 接近 0，即保持张开
TARGET_GRIPPER_STATE = 0.0 

# 移动速度 (弧度/秒)。保持慢速，确保安全。
MOVE_SPEED = 0.25 
# ===========================================

def close_gripper(arm, current_gripper_state):
    # 模拟夹爪关闭
    if current_gripper_state > 0.9: return 
    arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x2E, 0xE0])
    arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
    return 1.0

def open_gripper(arm, current_gripper_state):
    # 模拟夹爪打开
    if current_gripper_state < 0.1: return
    arm.getset_tgpio_modbus_data([0x01, 0x10, 0x01, 0x02, 0x00, 0x02, 0x04, 0x0, 0x0, 0x00, 0x00])
    arm.getset_tgpio_modbus_data([0x01, 0x06, 0x01, 0x08, 0x00, 0x01])
    return 0.0

def main():
    print(f"Connecting to xArm at {ROBOT_IP}...")
    arm = XArmAPI(ROBOT_IP)
    
    if arm.connected:
        print(">>> Connection successful.")
    else:
        print(">>> Failed to connect to xArm.")
        return

    # 1. 初始化和安全设置
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    # 切换到 Mode 0 (位置规划模式)
    arm.set_mode(0)
    arm.set_state(0)
    
    print("-" * 40)
    print(f"目标位置 (弧度): {np.array(TARGET_JOINTS_RAD, dtype=np.float32)}")
    print("-" * 40)
    
    confirm = input("❗ 请确认环境安全，按回车键开始移动 (或输入 'q' 取消): ")
    if confirm.strip().lower() == 'q':
        arm.disconnect()
        return

    # 2. 关节移动
    print(f">>> 机械臂开始以 {MOVE_SPEED} rad/s 移动到目标位置...")
    
    ret = arm.set_servo_angle(
        angle=TARGET_JOINTS_RAD, 
        speed=MOVE_SPEED, 
        is_radian=True, 
        wait=True  # 等待机械臂到达目标位置
    )

    if ret == 0:
        print("✅ 机械臂成功到达目标位置。")
        # 3. 夹爪移动
        print(f">>> 执行夹爪动作 ({'关闭' if TARGET_GRIPPER_STATE > 0.8 else '张开'})...")
        if TARGET_GRIPPER_STATE > 0.8:
            close_gripper(arm, 0.0)
        else:
            open_gripper(arm, 1.0)
    else:
        print(f"❌ 机械臂移动失败，错误代码: {ret}")
        print("请检查限位、电源或连接是否正常。")
        
    time.sleep(2)

    # 4. 清理
    print(">>> 测试完成，机械臂保持当前位置。")
    arm.set_state(4) # 保持状态
    arm.disconnect()

if __name__ == "__main__":
    main()