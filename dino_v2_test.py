# %%
# pip install torch torchvision timm opencv-python umap-learn matplotlib

# %%
from loguru import logger
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import cv2
import torch
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# model = timm.create_model(
#     "models/timm/vit_base_patch14_dinov2.lvd142m",
#     pretrained=True,
#     num_classes=0
# ).to(device).eval()
model = timm.create_model(
    "vit_base_patch14_dinov2.lvd142m",
    pretrained=False,                 # 关键：关掉在线下载
    num_classes=0,
    checkpoint_path="models/timm/vit_base_patch14_dinov2.lvd142m/model.safetensors"
).to(device).eval()

print(model)

# %%

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


@torch.no_grad()
def extract_dinov2_features(video_path, fps=2):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(video_fps / fps)

    t_start = 500  # seconds
    t_end = 600  # seconds
    frame_start = int(t_start * video_fps)
    frame_end = int(t_end * video_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    feats = []
    times = []

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

        t += 1.0 / video_fps
        if t > t_end:
            logger.debug(f'Reached {t_end=}')
            break

    cap.release()

    X = np.asarray(feats)
    times = np.asarray(times)
    print(f'{X.shape=}, {times.shape=}')
    return X, times


X, times = extract_dinov2_features("./video/ekaterina.mp4", fps=2)

reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.05,
    n_components=2,
    metric="euclidean",
    random_state=42
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
plt.show()

# %%
