# download_vae_mirror.py
import os
from huggingface_hub import snapshot_download

def download_vae_with_mirror():
    """使用镜像源下载VAE模型"""
    
    # 方法1: 设置环境变量使用镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 方法2: 直接使用镜像URL
    model_name = "stabilityai/sd-vae-ft-mse"
    model_name = "openai/clip-vit-base-patch32"
    model_name = "runwayml/stable-diffusion-v1-5"
    
    print("使用镜像源下载模型...")
    print(f"模型: {model_name}")
    
    try:
        # 使用镜像
        snapshot_download(
            repo_id=model_name,
            local_dir="./models/sd-vae-ft-mse",
            local_dir_use_symlinks=False,
            resume_download=True,
            endpoint="https://hf-mirror.com"  # 指定镜像
        )
        print("✅ 下载成功!")
        
    except Exception as e:
        print(f"❌ 镜像下载失败: {e}")
        print("\n尝试备用方案...")
        download_vae_backup()

def download_vae_backup():
    """备用下载方案"""
    print("\n=== 备用方案 ===")
    print("1. 使用预下载的文件")
    print("2. 使用较小的替代模型")
    print("3. 手动下载")
    
    choice = input("\n请选择方案 (1/2/3): ").strip()
    
    if choice == "1":
        use_pre_downloaded()
    elif choice == "2":
        download_smaller_model()
    elif choice == "3":
        manual_download_guide()
    else:
        print("无效选择，使用较小的替代模型")
        download_smaller_model()

def download_smaller_model():
    """下载较小的替代模型"""
    print("\n下载较小的模型: stabilityai/sdxl-vae")
    
    try:
        snapshot_download(
            repo_id="stabilityai/sdxl-vae",
            local_dir="./models/sdxl-vae",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✅ 较小模型下载成功!")
    except Exception as e:
        print(f"❌ 仍然失败: {e}")
        manual_download_guide()

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

def use_pre_downloaded():
    """使用预下载的文件（如果你已经有了）"""
    print("\n请将已有的模型文件放入 ./models/sd-vae-ft-mse/ 目录")
    print("然后创建文件检查脚本:")
    
    check_script = """
import os
import json

def check_model_files():
    model_dir = "./models/sd-vae-ft-mse"
    required_files = ["config.json", "model_index.json"]
    
    if not os.path.exists(model_dir):
        print("❌ 模型目录不存在")
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            print(f"❌ 缺少文件: {file}")
            return False
    
    # 检查模型文件
    model_files = [f for f in os.listdir(model_dir) 
                   if f.endswith(('.safetensors', '.bin', '.pth'))]
    
    if not model_files:
        print("❌ 没有找到模型权重文件")
        return False
    
    print("✅ 模型文件完整")
    return True

if __name__ == "__main__":
    check_model_files()
"""
    
    with open("check_model.py", "w") as f:
        f.write(check_script)
    
    print("已创建 check_model.py")
    print("运行: python check_model.py 检查文件")

if __name__ == "__main__":
    download_vae_with_mirror()