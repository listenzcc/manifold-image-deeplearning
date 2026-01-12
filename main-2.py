# 安装依赖: pip install diffusers transformers accelerate torch torchvision pillow numpy opencv-python scipy

import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from scipy import interpolate
import os

class StableDiffusionAutoEncoder:
    """使用Stable Diffusion预训练VAE进行图像编码和解码"""
    
    def __init__(self, model_name="models/stabilityai/sd-vae-ft-mse", device=None):
        """
        初始化Stable Diffusion VAE
        
        Args:
            model_name: VAE模型名称
                - "stabilityai/sd-vae-ft-mse" (推荐，高质量)
                - "stabilityai/sd-vae-ft-ema" (另一种高质量版本)
            device: 设备 ('cuda', 'cpu')，默认自动检测
        """
        from diffusers import AutoencoderKL
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        print(f"加载VAE模型: {model_name}...")
        
        # 加载预训练VAE
        self.vae = AutoencoderKL.from_pretrained(model_name)
        self.vae = self.vae.to(self.device)
        self.vae.eval()  # 设置为评估模式
        
        # VAE配置信息
        self.scaling_factor = self.vae.config.scaling_factor
        self.latent_channels = self.vae.config.latent_channels  # 通常是4
        print(f"VAE配置: scaling_factor={self.scaling_factor}, latent_channels={self.latent_channels}")
        
        # 图像变换
        self.setup_transforms()
    
    def setup_transforms(self):
        """设置图像预处理和后处理变换"""
        from torchvision import transforms
        
        # 编码时的预处理：图像 -> 张量
        self.encode_transform = transforms.Compose([
            transforms.Resize((512, 512)),  # SD VAE标准输入尺寸
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
        ])
        
        # 解码时的后处理：张量 -> 图像
        self.decode_transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),  # [-1,1] -> [0,1]
            transforms.Lambda(lambda t: t.clamp(0, 1)),  # 确保在[0,1]范围内
            transforms.Lambda(lambda t: t * 255.0),  # [0,1] -> [0,255]
        ])
    
    def encode_image(self, image_array):
        """
        将图像编码到潜在空间
        
        Args:
            image_array: numpy数组 (H, W, 3) 值范围0-255
            
        Returns:
            latent: numpy数组 (4, 64, 64) VAE的潜在表示
        """
        # 转换为PIL图像
        pil_image = Image.fromarray(image_array.astype('uint8'))
        
        # 预处理
        image_tensor = self.encode_transform(pil_image).unsqueeze(0).to(self.device)
        
        # VAE编码
        with torch.no_grad():
            posterior = self.vae.encode(image_tensor)
            latent = posterior.latent_dist.sample()
            latent = latent * self.scaling_factor
        
        # 返回numpy数组
        return latent.cpu().numpy()[0]
    
    def decode_latent(self, latent):
        """
        从潜在空间解码图像
        
        Args:
            latent: numpy数组 (4, 64, 64) 或 (4, H, W)
            
        Returns:
            image_array: numpy数组 (H, W, 3) 值范围0-255
        """
        # 确保是正确形状
        if len(latent.shape) == 4:
            latent = latent[0]  # 去掉批次维度
        
        # 转换为张量
        latent_tensor = torch.FloatTensor(latent).unsqueeze(0).to(self.device)
        
        # 应用缩放因子
        latent_tensor = latent_tensor / self.scaling_factor
        
        # VAE解码
        with torch.no_grad():
            image_tensor = self.vae.decode(latent_tensor).sample
        
        # 后处理
        image_tensor = self.decode_transform(image_tensor[0])
        
        # 转换为numpy数组并调整通道顺序
        image_array = image_tensor.cpu().numpy().transpose(1, 2, 0)
        image_array = np.clip(image_array, 0, 255).astype('uint8')
        
        # 调整到目标尺寸200x200
        pil_image = Image.fromarray(image_array)
        pil_image = pil_image.resize((200, 200), Image.Resampling.LANCZOS)
        
        return np.array(pil_image)
    
    def encode_images_batch(self, image_dir):
        """
        批量编码目录中的所有图像
        
        Args:
            image_dir: 图像目录路径
            
        Returns:
            images_list: 原始图像列表
            latents_list: 潜在向量列表
        """
        image_dir = Path(image_dir)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        image_files = [f for f in image_dir.iterdir() if f.suffix in image_extensions]

        # Limit the images to 10 images.
        image_files = image_files[:10]
        
        if not image_files:
            raise ValueError(f"在目录 {image_dir} 中没有找到图像文件")
        
        images_list = []
        latents_list = []
        
        print(f"找到 {len(image_files)} 张图像，开始编码...")
        
        for i, img_file in enumerate(image_files):
            try:
                # 加载图像
                img = Image.open(img_file).convert('RGB')
                img_array = np.array(img)
                
                # 编码到潜在空间
                latent = self.encode_image(img_array)
                
                images_list.append(img_array)
                latents_list.append(latent)
                
                if (i + 1) % 10 == 0:
                    print(f"  已处理 {i+1}/{len(image_files)} 张图像")
                    
            except Exception as e:
                print(f"  警告: 无法处理图像 {img_file.name}: {e}")
                continue
        
        print(f"编码完成: {len(latents_list)} 张图像成功编码")
        return images_list, latents_list


class ManifoldExpander:
    """在潜在空间进行流形扩展"""
    
    def __init__(self, interpolation_method='cubic'):
        """
        Args:
            interpolation_method: 插值方法 ('linear', 'cubic', 'quadratic')
        """
        self.interpolation_method = interpolation_method
    
    def expand_in_latent_space(self, latents, k=5):
        """
        在潜在空间进行流形扩展
        
        Args:
            latents: 潜在向量列表，每个形状为 (C, H, W)
            k: 扩展倍数
            
        Returns:
            expanded_latents: 扩展后的潜在向量
        """
        n = len(latents)
        if n < 2:
            raise ValueError("至少需要2张图像进行插值")
        
        # 将潜在向量展平以便插值
        latents_flat = []
        for latent in latents:
            latents_flat.append(latent.flatten())
        
        latents_flat = np.array(latents_flat)  # (n, latent_dim)
        latent_dim = latents_flat.shape[1]
        
        # 原始点
        x_original = np.arange(n)
        
        # 扩展点
        n_expanded = n * k
        x_expanded = np.linspace(0, n-1, n_expanded)
        
        # 插值
        expanded_flat = np.zeros((n_expanded, latent_dim))
        
        print(f"在潜在空间进行 {self.interpolation_method} 插值...")
        print(f"  原始: {n} 个点 -> 扩展: {n_expanded} 个点")
        print(f"  潜在维度: {latent_dim}")
        
        # 对每个维度进行插值
        for i in range(latent_dim):
            if self.interpolation_method == 'cubic' and n >= 4:
                # 三次样条插值需要至少4个点
                f = interpolate.CubicSpline(x_original, latents_flat[:, i])
            else:
                # 线性插值
                f = interpolate.interp1d(x_original, latents_flat[:, i], 
                                        kind='linear', fill_value="extrapolate")
            expanded_flat[:, i] = f(x_expanded)
        
        # 将展平的潜在向量恢复为原始形状
        expanded_latents = []
        latent_shape = latents[0].shape
        
        for i in range(n_expanded):
            latent_reshaped = expanded_flat[i].reshape(latent_shape)
            expanded_latents.append(latent_reshaped)
        
        return expanded_latents


class VideoGenerator:
    """视频生成器"""
    
    @staticmethod
    def create_video(images, output_path, fps=30, quality=90):
        """
        从图像序列创建视频
        
        Args:
            images: 图像列表，每个为numpy数组 (H, W, 3) 0-255
            output_path: 输出视频路径
            fps: 帧率
            quality: 视频质量 (1-100)
        """
        if not images:
            raise ValueError("没有图像可以创建视频")
        
        # 获取图像尺寸
        height, width = images[0].shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"创建视频: {output_path}")
        print(f"  尺寸: {width}x{height}, 帧率: {fps}, 总帧数: {len(images)}")
        
        for i, img in enumerate(images):
            # 确保图像是BGR格式
            if img.shape[2] == 3:
                frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                frame = img
            
            video.write(frame)
            
            if (i + 1) % 50 == 0:
                print(f"  已写入 {i+1}/{len(images)} 帧")
        
        video.release()
        print(f"视频创建完成: {output_path}")


def process_image_manifold(image_dir, output_video, k=5, fps=30, vae_model="models/stabilityai/sd-vae-ft-mse"):
    """
    完整的图像流形处理流程
    
    Args:
        image_dir: 输入图像目录
        output_video: 输出视频路径
        k: 扩展倍数
        fps: 视频帧率
        vae_model: VAE模型名称
    """
    print("=" * 60)
    print("图像流形扩展处理器")
    print("=" * 60)
    
    # 1. 初始化VAE编码器
    print("\n[步骤 1/4] 初始化VAE编码器...")
    vae_encoder = StableDiffusionAutoEncoder(model_name=vae_model)
    
    # 2. 编码所有图像
    print("\n[步骤 2/4] 编码图像到潜在空间...")
    original_images, original_latents = vae_encoder.encode_images_batch(image_dir)
    
    if len(original_latents) < 2:
        raise ValueError(f"至少需要2张图像，但只成功加载了{len(original_latents)}张")
    
    # 3. 在潜在空间进行流形扩展
    print("\n[步骤 3/4] 在潜在空间进行流形扩展...")
    expander = ManifoldExpander(interpolation_method='cubic')
    expanded_latents = expander.expand_in_latent_space(original_latents, k=k)
    
    # 4. 解码扩展后的潜在向量
    print("\n[步骤 4/4] 解码为图像序列...")
    expanded_images = []
    
    for i, latent in enumerate(expanded_latents):
        try:
            decoded_image = vae_encoder.decode_latent(latent)
            expanded_images.append(decoded_image)
            
            if (i + 1) % 20 == 0:
                print(f"  已解码 {i+1}/{len(expanded_latents)} 帧")
                
        except Exception as e:
            print(f"  警告: 解码第{i}帧时出错: {e}")
            # 使用黑色图像作为替代
            expanded_images.append(np.zeros((200, 200, 3), dtype=np.uint8))
    
    # 5. 创建视频
    print("\n[视频生成] 创建MP4视频...")
    VideoGenerator.create_video(expanded_images, output_video, fps=fps)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"输入图像: {len(original_images)} 张")
    print(f"扩展倍数: {k} 倍")
    print(f"输出帧数: {len(expanded_images)} 帧")
    print(f"视频时长: {len(expanded_images)/fps:.2f} 秒")
    print(f"输出文件: {output_video}")
    print("=" * 60)
    
    return expanded_images


# 简化使用版本
def quick_process(image_dir, k=5, fps=30, output="output_video.mp4"):
    """
    快速处理函数
    
    Args:
        image_dir: 图像目录
        k: 扩展倍数 (默认5)
        fps: 帧率 (默认30)
        output: 输出视频文件名 (默认"output_video.mp4")
    """
    print("开始快速处理...")
    
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误: 目录 '{image_dir}' 不存在")
        return
    
    # 处理
    try:
        process_image_manifold(
            image_dir=image_dir,
            output_video=output,
            k=k,
            fps=fps
        )
        print(f"\n✅ 处理完成! 视频已保存为: {output}")
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")

def main_with_cli():
    import argparse
    
    parser = argparse.ArgumentParser(description="图像流形扩展视频生成器")
    parser.add_argument("--image_dir", type=str, required=True, 
                       help="输入图像目录路径")
    parser.add_argument("--output", type=str, default="manifold_video.mp4",
                       help="输出视频路径 (默认: manifold_video.mp4)")
    parser.add_argument("--k", type=int, default=5,
                       help="扩展倍数 (默认: 5)")
    parser.add_argument("--fps", type=int, default=30,
                       help="视频帧率 (默认: 30)")
    parser.add_argument("--model", type=str, default="stabilityai/sd-vae-ft-mse",
                       choices=["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"],
                       help="VAE模型 (默认: stabilityai/sd-vae-ft-mse)")
    
    args = parser.parse_args()

    # 使用命令行参数
    process_image_manifold(
        image_dir=args.image_dir,
        output_video=args.output,
        k=args.k,
        fps=args.fps,
        vae_model=args.model
    )

# 命令行接口
if __name__ == "__main__":
    # 快速使用示例
    if True:  # 设为True直接运行示例
        quick_process(
            image_dir="./DesktopPictures",  # 你的图像目录
            k=30,
            fps=30,
            output="my_video.mp4"
        )
    else:
        main_with_cli()