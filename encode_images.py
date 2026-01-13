"""
File: main.py
Author: Chuncheng Zhang
Date: 2026-01-12
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
Image
  ↓ CLIP image encoder
z ∈ R^512
  ↓ linear interpolation
z(t)
  ↓ diffusion (image-conditional, no text)
Image(t)
  ↓
Video


Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""

# %% ---- 2026-01-12 ------------------------
# Requirements and constants
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel

# %% ---- 2026-01-12 ------------------------
# Function and class


class CLIPImageEncoder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = (
            CLIPModel.from_pretrained("models/openai/clip-vit-base-patch32")
            .to(self.device)
            .eval()
        )

        self.processor = CLIPProcessor.from_pretrained("models/openai/clip-vit-base-patch32")

    @torch.no_grad()
    def encode(self, image_np):
        inputs = self.processor(
            images=Image.fromarray(image_np), return_tensors="pt"
        ).to(self.device)

        z = self.model.get_image_features(**inputs)
        z = z / z.norm(dim=-1, keepdim=True)  # normalize
        return z[0].cpu().numpy()  # (512,)


def interpolate_clip(z_list, k=20):
    """
    z_list: [z1, z2, ..., zn], each (512,)
    return: list of interpolated z
    """
    z_list = np.stack(z_list)  # (n, 512)

    out = []
    for i in range(len(z_list) - 1):
        for t in np.linspace(0, 1, k, endpoint=False):
            z = (1 - t) * z_list[i] + t * z_list[i + 1]
            z = z / np.linalg.norm(z)
            out.append(z)

    out.append(z_list[-1])
    return out


from diffusers import StableDiffusionPipeline


class CLIPDiffusionGenerator:
    def __init__(self, device=None, steps=30):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "models/runwayml/stable-diffusion-v1-5",
            dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)

        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False

    @torch.no_grad()
    def generate(self, clip_vector, seed=None):
        """
        clip_vector: (512,)
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        clip_emb = torch.tensor(clip_vector, device=self.device)
        clip_emb = clip_emb.unsqueeze(0).unsqueeze(1)  # (1,1,512)

        image = self.pipe(
            prompt_embeds=clip_emb,
            num_inference_steps=self.steps,
            guidance_scale=1.0,
            generator=generator,
        ).images[0]

        return np.array(image)


import cv2


def write_video(images, path, fps=24):
    h, w = images[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for img in images:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    writer.release()


from pathlib import Path


def clip_manifold_video(image_dir):
    image_dir = Path(image_dir)
    files = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
        ]
    )

    assert len(files) >= 2, "至少需要两张图像"

    encoder = CLIPImageEncoder()
    generator = CLIPDiffusionGenerator(steps=25)

    print("CLIP 编码...")
    zs = []
    for f in tqdm(files, 'Encode images'):
        img = np.array(Image.open(f).convert("RGB"))
        zs.append(encoder.encode(img))

    print(np.array(zs).shape)

    encoded = np.array(zs)

    print(f"完成: {encoded.shape=}")
    np.save('encoded.npy', encoded)
    with open('encoded.info', 'w') as f:
        f.writelines([f'{p}\n' for p in files])
        


# %% ---- 2026-01-12 ------------------------
# Play ground
if __name__ == "__main__":
    clip_manifold_video(
        image_dir="./extracted_frames",
    )


# %% ---- 2026-01-12 ------------------------
# Pending


# %% ---- 2026-01-12 ------------------------
# Pending
