# extract_frames_basic.py
import cv2
import os

def extract_frames_basic(video_path, output_dir="frames", prefix="frame"):
    """
    提取视频的所有帧并保存为图像
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        prefix: 图像文件名前缀
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"  文件: {video_path}")
    print(f"  分辨率: {width}×{height}")
    print(f"  帧率: {fps:.2f} fps")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {total_frames/fps:.2f} 秒")
    print(f"输出目录: {output_dir}")
    print("-" * 40)
    
    frame_count = 0
    success = True
    
    print("开始提取帧...")
    while success:
        # 读取一帧
        success, frame = cap.read()
        
        if not success:
            break
        
        # 生成文件名
        filename = f"{prefix}_{frame_count:06d}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # 保存帧为图像
        cv2.imwrite(filepath, frame)
        
        frame_count += 1
        
        # 显示进度
        if frame_count % 50 == 0:
            print(f"  已提取 {frame_count}/{total_frames} 帧")
    
    # 释放资源
    cap.release()
    
    print("-" * 40)
    print(f"✅ 完成!")
    print(f"  成功提取 {frame_count} 帧")
    print(f"  图像保存在: {output_dir}")
    
    return frame_count

if __name__ == "__main__":
    # 使用示例
    video_file = "my_video.mp4"  # 修改为你的视频文件
    output_folder = "extracted_frames"
    
    extract_frames_basic(video_file, output_folder)