"""
File: download_hf_model.py
Author: Chuncheng Zhang
Date: 2026-01-12
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Download hf model.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-01-12 ------------------------
# Requirements and constants
import os
from huggingface_hub import snapshot_download


# %% ---- 2026-01-12 ------------------------
# Function and class

def download_hf_model_with_mirror(model_name):
    """使用镜像源下载VAE模型"""

    # 设置环境变量使用镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    print("使用镜像源下载模型...")
    print(f"模型: {model_name}")

    # 使用镜像
    snapshot_download(
        repo_id=model_name,
        local_dir=f"./models/{model_name}",
        # local_dir_use_symlinks=False,
        resume_download=True,
        endpoint="https://hf-mirror.com"  # 指定镜像
    )
    print("✅ 下载成功!")


def manual_download_guide():
    """手动下载指南"""
    print("\n" + "="*60)
    print("手动下载指南")
    print("="*60)
    print("\n请按照以下步骤操作:")
    print("\n1. 访问 HuggingFace 镜像站:")
    print("   https://hf-mirror.com/stabilityai/sd-vae-ft-mse")
    print("\n2. 点击 'Files and versions' 标签页")
    print("\n3. 下载以下文件:")
    print("   - config.json")
    print("   - model_index.json")
    print("   - diffusion_pytorch_model.safetensors")
    print("\n4. 将文件保存到: ./models/sd-vae-ft-mse/")
    print("\n5. 确保目录结构如下:")
    print("   ./models/sd-vae-ft-mse/")
    print("   ├── config.json")
    print("   ├── model_index.json")
    print("   └── diffusion_pytorch_model.safetensors")
    print("\n完成后重新运行程序。")


if __name__ == "__main__":
    # model_name = "stabilityai/sd-vae-ft-mse"
    # model_name = "openai/clip-vit-base-patch32"
    model_name = "runwayml/stable-diffusion-v1-5"
    model_name = 'timm/vit_base_patch14_dinov2.lvd142m'
    download_hf_model_with_mirror(model_name)

# %% ---- 2026-01-12 ------------------------
# Play ground


# %% ---- 2026-01-12 ------------------------
# Pending


# %% ---- 2026-01-12 ------------------------
# Pending
