import cv2
import numpy as np

def test_fixed_cameras(scale_factor=0.5):
    """
    使用 Udev 固定的设备路径读取三个相机画面
    """
    # 定义 Udev 创建的固定设备路径
    camera_configs = [
        {"name": "High", "path": "/dev/cam_high"},
        {"name": "Left Wrist", "path": "/dev/cam_left_wrist"},
        {"name": "Right Wrist", "path": "/dev/cam_right_wrist"}
    ]

    caps = []
    # 原始分辨率设定
    ORIG_WIDTH, ORIG_HEIGHT = 1280, 720
    # 缩放后的单图分辨率
    SCALED_WIDTH = int(ORIG_WIDTH * scale_factor)
    SCALED_HEIGHT = int(ORIG_HEIGHT * scale_factor)

    print("--- 正在初始化固定路径相机 ---")
    
    for config in camera_configs:
        path = config["path"]
        name = config["name"]
        
        # OpenCV 的 VideoCapture 可以直接接收字符串路径
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, ORIG_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ORIG_HEIGHT)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            caps.append({"cap": cap, "name": name, "status": True})
            print(f"✅ 成功连接到 {name}: {path}")
        else:
            caps.append({"cap": None, "name": name, "status": False})
            print(f"❌ 无法连接到 {name}: {path} (请检查 Docker 挂载或 Udev 规则)")

    if not any(c["status"] for c in caps):
        print("\n错误: 所有相机均不可用！请确认宿主机执行过 udevadm trigger 且 Docker 已挂载设备。")
        return

    print(f"\n🎥 开始预览。分辨率: {SCALED_WIDTH}x{SCALED_HEIGHT} | 退出请按 'q'")

    try:
        while True:
            frames_to_show = []
            
            for item in caps:
                if item["status"]:
                    ret, frame = item["cap"].read()
                    if ret:
                        # 缩放
                        frame_small = cv2.resize(frame, (SCALED_WIDTH, SCALED_HEIGHT))
                        # 标注名称
                        cv2.putText(frame_small, item["name"], (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        frames_to_show.append(frame_small)
                        continue
                
                # 如果读取失败或状态不对，补黑色块
                black_frame = np.zeros((SCALED_HEIGHT, SCALED_WIDTH, 3), dtype=np.uint8)
                cv2.putText(black_frame, f"{item['name']} (OFFLINE)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                frames_to_show.append(black_frame)

            # 水平拼接三个画面
            combined = cv2.hconcat(frames_to_show)
            
            cv2.imshow('xArm Camera Test (Fixed Udev Path)', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 释放资源
        for item in caps:
            if item["cap"] is not None:
                item["cap"].release()
        cv2.destroyAllWindows()
        print("🔌 相机已释放。")

if __name__ == "__main__":
    test_fixed_cameras(scale_factor=0.4)