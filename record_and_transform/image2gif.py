import os
import glob
from PIL import Image

def images_to_gif(image_dir, output_gif_path, frame_duration=100, image_formats=('jpg', 'png', 'jpeg')):
    """
    将指定目录下的图片按顺序转换为GIF动图
    
    参数:
        image_dir: 图片目录路径
        output_gif_path: 输出GIF的路径（如 /home/openpi/result.gif）
        frame_duration: 每一帧的持续时间（毫秒），值越小播放越快
        image_formats: 支持的图片格式
    """
    # 1. 验证目录是否存在
    if not os.path.isdir(image_dir):
        print(f"错误：目录不存在 -> {image_dir}")
        return False
    
    # 2. 获取所有图片文件路径
    image_paths = []
    for fmt in image_formats:
        # 匹配所有指定格式的图片
        paths = glob.glob(os.path.join(image_dir, f'*.{fmt}'))
        image_paths.extend(paths)
    
    if not image_paths:
        print(f"错误：目录下未找到任何图片 -> {image_dir}")
        return False
    
    # 3. 按文件名数字排序（关键：保证播放顺序正确）
    def extract_number(filename):
        """从文件名中提取数字用于排序"""
        try:
            # 提取文件名中的数字部分（适配如 img_123.jpg、123.png 等格式）
            name = os.path.splitext(os.path.basename(filename))[0]
            # 过滤出所有数字字符并转为整数
            num = int(''.join(filter(str.isdigit, name)))
            return num
        except:
            # 无法提取数字则返回0，排到最前面
            return 0
    
    # 按数字大小排序
    image_paths.sort(key=extract_number)
    print(f"找到 {len(image_paths)} 张图片，已按顺序排序")
    
    # 4. 读取图片并转换为GIF
    frames = []
    for idx, img_path in enumerate(image_paths):
        try:
            # 打开图片并转为RGB模式（避免透明通道问题）
            img = Image.open(img_path).convert('RGB')
            frames.append(img)
            # 打印进度（可选）
            if (idx + 1) % 10 == 0:
                print(f"已加载 {idx + 1}/{len(image_paths)} 张图片")
        except Exception as e:
            print(f"警告：跳过损坏的图片 {img_path}，错误：{e}")
            continue
    
    if not frames:
        print("错误：没有可用的图片来生成GIF")
        return False
    
    # 5. 保存GIF动图
    try:
        # 第一帧作为基础，后续帧追加，设置循环播放（0表示无限循环）
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True  # 优化GIF大小
        )
        print(f"成功生成GIF动图 -> {output_gif_path}")
        print(f"GIF信息：共 {len(frames)} 帧，每帧时长 {frame_duration} 毫秒")
        return True
    except Exception as e:
        print(f"错误：保存GIF失败，{e}")
        return False

if __name__ == "__main__":
    # ========== 配置参数（根据需要修改） ==========
    # 目标图片目录（你指定的路径）
    IMAGE_DIR = "/home/openpi/data/data_raw/test/raw/episode_72/images/cam_left_wrist"
    # 输出GIF路径（建议保存在易访问的位置）
    OUTPUT_GIF = "./1.gif"
    # 每帧时长（毫秒），默认100ms，值越小播放越快
    FRAME_DURATION = 100
    
    # 禁用Qt图形界面（避免你之前遇到的XCB报错）
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    
    # 执行转换
    success = images_to_gif(IMAGE_DIR, OUTPUT_GIF, FRAME_DURATION)
    if success:
        print(f"\nGIF生成完成！文件位置：{OUTPUT_GIF}")
    else:
        print("\nGIF生成失败，请检查路径和图片！")
