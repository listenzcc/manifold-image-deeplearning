# %%
# pip install torch torchvision timm opencv-python umap-learn matplotlib

# %%
import cv2
import umap
import timm
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from torchvision import transforms


# %%


@torch.no_grad()
def extract_dinov2_features(video_path, t_start, t_end, fps=2):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(video_fps / fps)

    frame_start = int(t_start * video_fps)
    frame_end = int(t_end * video_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    feats = []
    times = []
    idxs = []

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    t = 0.0
    for idx in tqdm(range(frame_start, frame_end)):
        ret, frame = cap.read()
        if not ret:
            logger.error('Falied on read frame!')
            break

        if idx % step == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = preprocess(img).unsqueeze(0).to(device)

            # forward
            tokens = model.forward_features(img)  # ["x_norm_patchtokens"]
            # print(f'{tokens.shape}')
            # tokens: [1, N, 768]

            mean = tokens.mean(dim=1)
            std = tokens.std(dim=1)

            x = torch.cat([mean, std], dim=1)  # [1, 1536]
            feats.append(x.cpu().numpy()[0])
            times.append(t)
            idxs.append(idx)

        t += 1.0 / video_fps
        if t > t_end:
            logger.debug(f'Reached {t_end=}')
            break

    cap.release()

    X = np.asarray(feats)
    times = np.asarray(times)
    print(f'{X.shape=}, {times.shape=}')
    return X, times, idxs


# %%
'''
Generate new video
Args:
    - Y: The 2D projection of DINO features of the key frames
    - idxs: The indexes of the key frames
    - times: The times of the key frames
    - t_start, t_end: The start and end time of the video.
'''


def normalize_2d(Y, eps=1e-6):
    Y_min = Y.min(axis=0)
    Y_max = Y.max(axis=0)
    return (Y - Y_min) / (Y_max - Y_min + eps)


def render_manifold_video(
    video_path,
    output_path,
    Y,
    idxs,
    t_start,
    t_end,
    point_radius=3,
    highlight_radius=6
):
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_start = int(t_start * video_fps)
    frame_end = int(t_end * video_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (W, H)
    )

    # 右上角 1/4 区域（正方形）
    box_size = min(W, H) // 2
    box_x0 = W - box_size
    box_y0 = 0

    # 归一化 manifold
    Yn = normalize_2d(Y)

    # 预计算像素坐标
    pts = []
    for y in Yn:
        px = int(box_x0 + y[0] * box_size)
        py = int(box_y0 + (1 - y[1]) * box_size)
        pts.append((px, py))

    for frame_idx in tqdm(range(frame_start, frame_end)):
        ret, frame = cap.read()
        if not ret:
            break

        # 当前视频帧时间
        t = frame_idx / video_fps
        if t > t_end:
            break

        # 更新当前关键帧指针
        key_ptr = len([e for e in idxs if e < frame_idx]) % len(pts)
        # while key_ptr + 1 < len(idxs) and idxs[key_ptr + 1] <= frame_idx:
        #     key_ptr += 1

        # ---- 画背景框 ----
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (box_x0, box_y0),
            (box_x0 + box_size, box_y0 + box_size),
            (40, 40, 40),
            thickness=-1
        )
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # ---- 画所有点 ----
        for i, (px, py) in enumerate(pts):
            cv2.circle(
                frame,
                (px, py),
                point_radius,
                (180, 180, 180),
                -1
            )

        # ---- 高亮当前点 ----
        px, py = pts[key_ptr]
        cv2.circle(
            frame,
            (px, py),
            highlight_radius,
            (0, 255, 255),
            -1
        )

        writer.write(frame)

    cap.release()
    writer.release()
    print('Done.')


# %%
# Setup
src_video = Path('./video/ekaterina.mp4')
t_start = 500  # seconds
t_end = 600  # seconds
dst_video = src_video.with_name(
    f'{src_video.name}.{t_start}-{t_end}.manifold.mp4')
dst_fig = src_video.with_name(
    f'{src_video.name}.{t_start}-{t_end}.manifold.png')


# %%
# Deep learning model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = timm.create_model(
    "vit_base_patch14_dinov2.lvd142m",
    pretrained=False,                 # 关键：关掉在线下载
    num_classes=0,
    checkpoint_path="models/timm/vit_base_patch14_dinov2.lvd142m/model.safetensors"
).to(device).eval()

print(model)

# %%
# Extract features
X, times, idxs = extract_dinov2_features(src_video, t_start, t_end, fps=2)

# %%
# Projection
reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.05,
    n_components=2,
    metric="euclidean",
    # random_state=42
)

Y = reducer.fit_transform(X)

plt.figure(figsize=(8, 6))
sc = plt.scatter(
    Y[:, 0], Y[:, 1],
    c=times,
    cmap="viridis",
    s=12
)
plt.plot(Y[:, 0], Y[:, 1], alpha=0.1)
plt.colorbar(sc, label="Time (s)")
plt.title("Movie Manifold (DINOv2 patch tokens)")
plt.axis("equal")
plt.tight_layout()
plt.savefig(dst_fig)

# %%
# Render new video with manifold
render_manifold_video(
    video_path=src_video,
    output_path=dst_video,
    Y=Y,
    idxs=idxs,
    t_start=t_start,
    t_end=t_end
)

# %%

# %%
