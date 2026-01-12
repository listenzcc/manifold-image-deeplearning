# 安装依赖
# pip install torch torchvision diffusers transformers accelerate

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from torchvision.transforms import ToTensor
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载 Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(device)

# 2. 假设你有一个 CLIP 512 维向量
clip_vector = torch.randn(1, 512).to(device)  # 这里替换成你的真实向量

# 3. 将 CLIP 向量映射为潜向量 z
# Stable Diffusion 的 VAE 潜空间通常是 [B, 4, H/8, W/8]，我们可以用随机初始化并优化
latent_shape = (1, 4, 64, 64)  # 假设生成 512x512 图像
latents = torch.randn(latent_shape, device=device, requires_grad=True)

optimizer = torch.optim.Adam([latents], lr=0.05)
num_steps = 200

for step in range(num_steps):
    optimizer.zero_grad()
    
    # 解码潜向量为图像
    img = pipe.vae.decode(latents).sample  # [B, 3, H, W]
    img_norm = (img / 2 + 0.5).clamp(0, 1)
    
    # 用 CLIP 编码图像
    image_embeddings = pipe.text_encoder(img_norm.flatten(2)).last_hidden_state.mean(dim=1)
    
    # 计算损失（目标 CLIP 向量 - 图像 CLIP 向量）
    loss = torch.nn.functional.mse_loss(image_embeddings, clip_vector)
    loss.backward()
    optimizer.step()
    
    if step % 20 == 0:
        print(f"Step {step}, loss {loss.item()}")

# 4. 最终解码生成图像
final_img = pipe.vae.decode(latents).sample
final_img = (final_img / 2 + 0.5).clamp(0, 1)  # 归一化到 0~1

# 转为 PIL 显示
from torchvision.transforms import ToPILImage
ToPILImage()(final_img[0].cpu())
