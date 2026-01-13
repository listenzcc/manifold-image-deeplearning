import cv2
import os
import argparse
from datetime import datetime

def extract_frames_from_time(video_path, start_time_seconds, dt=0, total_seconds=10, output_dir="extracted_frames"):
    """
    从指定起始时间开始提取10秒内的视频帧，可设置采样间隔
    
    参数:
        video_path: 视频文件路径
        start_time_seconds: 起始时间（秒）
        dt: 采样间隔（秒），0表示提取所有帧
        total_seconds: 采集多长时间
        output_dir: 输出目录
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    fps_int = int(round(fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    duration = total_frames / fps  # 视频总时长（秒）
    
    print(f"视频信息：")
    print(f"  - 帧率: {fps:.2f} FPS")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 总时长: {duration:.2f} 秒")
    if dt > 0:
        print(f"  - 采样间隔: {dt} 秒")
    
    # 计算要提取的时间范围
    end_time = start_time_seconds + total_seconds  # 提取10秒
    
    # 转换为帧号
    start_frame = int(start_time_seconds * fps)
    end_frame = int(end_time * fps)
    
    # 确保范围在有效范围内
    if start_frame >= total_frames:
        print(f"错误：起始时间 {start_time_seconds} 秒超出了视频时长")
        return
    
    end_frame = min(end_frame, total_frames)
    actual_end_time = end_frame / fps
    
    print(f"\n提取范围：")
    print(f"  - 起始时间: {start_time_seconds:.2f} 秒 (第 {start_frame} 帧)")
    print(f"  - 结束时间: {actual_end_time:.2f} 秒 (第 {end_frame} 帧)")
    print(f"  - 时间长度: {actual_end_time - start_time_seconds:.2f} 秒")
    
    if dt == 0:
        # 提取所有帧
        frames_to_extract = list(range(start_frame, end_frame))
        total_extract_count = end_frame - start_frame
        print(f"  - 提取模式: 所有帧")
        print(f"  - 总共提取: {total_extract_count} 帧")
    else:
        # 按间隔采样
        frames_to_extract = []
        current_time = start_time_seconds
        
        while current_time < actual_end_time:
            frame_num = int(current_time * fps)
            if start_frame <= frame_num < end_frame:
                frames_to_extract.append(frame_num)
            current_time += dt
        
        total_extract_count = len(frames_to_extract)
        print(f"  - 提取模式: 间隔 {dt} 秒采样")
        print(f"  - 采样次数: {total_extract_count} 次")
        print(f"  - 采样时间点: {[start_time_seconds + i*dt for i in range(total_extract_count)]}")
    
    # 按顺序提取帧
    saved_count = 0
    
    for i, frame_num in enumerate(frames_to_extract):
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        ret, frame = cap.read()
        
        if not ret:
            print(f"警告：在帧 {frame_num} 读取失败")
            continue
        
        # 计算当前时间戳
        timestamp = start_time_seconds + (frame_num - start_frame) / fps
        
        # 生成文件名
        if dt > 0:
            filename = f"frame_{i+1:03d}_orig{frame_num:06d}_time_{timestamp:.3f}s.jpg"
        else:
            filename = f"frame_{frame_num:06d}_time_{timestamp:.3f}s.jpg"
        
        filepath = os.path.join(output_dir, filename)
        
        # 保存帧
        cv2.imwrite(filepath, frame)
        saved_count += 1
        
        # 显示进度
        if dt == 0 and frame_num % fps_int == 0:  # 每秒显示一次进度
            print(f"正在处理: {timestamp:.1f}/{actual_end_time:.1f} 秒")
        elif dt > 0:
            print(f"采样 {i+1}/{total_extract_count}: {timestamp:.3f} 秒")
    
    # 释放资源
    cap.release()
    
    print(f"\n提取完成！共保存了 {saved_count} 帧图像到目录: {output_dir}")
    
    # 计算实际采样率
    if dt > 0 and total_extract_count > 1:
        actual_dt = (actual_end_time - start_time_seconds) / (total_extract_count - 1)
        actual_fps = 1.0 / dt if dt > 0 else fps
        print(f"实际采样间隔: {actual_dt:.3f} 秒")
        print(f"实际采样率: {actual_fps:.2f} FPS")
    
    # 生成信息文件
    info_file = os.path.join(output_dir, "extraction_info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"提取信息 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"视频文件: {video_path}\n")
        f.write(f"视频帧率: {fps:.2f} FPS\n")
        f.write(f"视频时长: {duration:.2f} 秒\n")
        f.write(f"起始时间: {start_time_seconds:.2f} 秒\n")
        f.write(f"提取时长: 10 秒\n")
        f.write(f"结束时间: {actual_end_time:.2f} 秒\n")
        f.write(f"采样间隔: {dt if dt > 0 else '所有帧'} 秒\n")
        f.write(f"起始帧号: {start_frame}\n")
        f.write(f"结束帧号: {end_frame}\n")
        f.write(f"提取帧数: {saved_count}\n")
        
        if dt > 0 and len(frames_to_extract) > 0:
            f.write(f"\n采样时间点:\n")
            for i, frame_num in enumerate(frames_to_extract):
                timestamp = start_time_seconds + (frame_num - start_frame) / fps
                f.write(f"  {i+1:3d}. 时间: {timestamp:.3f}s, 帧号: {frame_num}\n")
    
    print(f"提取信息已保存到: {info_file}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='从视频中提取指定起始时间的10秒内的视频帧，可设置采样间隔',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python extract_frames.py video.mp4 30            # 从30秒开始提取10秒的所有帧
  python extract_frames.py video.mp4 30 -d 0.5     # 每隔0.5秒提取一帧
  python extract_frames.py video.mp4 30 -d 2       # 每隔2秒提取一帧
  python extract_frames.py video.mp4 30 -t 100 -d 2       # 采集100秒时长，每隔2秒提取一帧
  python extract_frames.py video.mp4 60 -o my_frames -d 1

        """
    )
    
    parser.add_argument('video_path', help='视频文件路径')
    parser.add_argument('start_time', type=float, help='起始时间（秒）')
    parser.add_argument('-d', '--dt', type=float, default=0,
                       help='采样间隔（秒），0表示提取所有帧（默认: 0）')
    parser.add_argument('-t', '--total', type=float, default=10,
                       help='采样多长时间（秒）')
    parser.add_argument('-o', '--output', default='extracted_frames', 
                       help='输出目录（默认: extracted_frames）')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.start_time < 0:
        print("错误：起始时间不能为负数")
        return
    
    if args.dt < 0:
        print("错误：采样间隔不能为负数")
        return
    
    # 提取帧
    extract_frames_from_time(args.video_path, args.start_time, args.dt, args.total, args.output)

if __name__ == "__main__":
    main()