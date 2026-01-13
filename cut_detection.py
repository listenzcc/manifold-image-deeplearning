# %%
# pip install umap-learn
# pip install pydiffmap

# %%
import cv2
import umap
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from loguru import logger
from pydiffmap import diffusion_map as dm


def fft_feature(frame, resize):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize)

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    power = np.log1p(np.abs(fft_shift))

    return power.flatten()


prev_gray = None


def optical_flow_feature(frame, resize):
    global prev_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize)

    if prev_gray is None:
        prev_gray = gray
        return None

    grid = 8
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # --- 分块统计（保留粗空间结构） ---
    h, w = mag.shape
    gh, gw = h // grid, w // grid

    feat = []
    for i in range(grid):
        for j in range(grid):
            m = mag[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            a = ang[i*gh:(i+1)*gh, j*gw:(j+1)*gw]

            feat.extend([
                m.mean(),
                m.std(),
                np.mean(np.cos(a)),
                np.mean(np.sin(a))
            ])

    prev_gray = gray

    return feat


def extract_fft_features(
    video_path,
    fps=2,
    resize=(128, 128)
):

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(video_fps / fps)

    t_start = 500  # seconds
    t_end = 1000  # seconds
    frame_start = int(t_start * video_fps)
    frame_end = int(t_end * video_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    logger.debug(f'Cap from {t_start=}, {frame_start=}')

    print(frame_start, frame_end)

    feats = []
    times = []

    t = 0.0
    for idx in tqdm(range(frame_start, frame_end)):
        ret, frame = cap.read()
        if not ret:
            logger.error('Falied on read frame!')
            break

        if idx % step == 0:
            f = fft_feature(frame, resize)
            # f = optical_flow_feature(frame, resize)
            if f is None:
                continue
            feats.append(f)
            times.append(t)

        t += 1.0 / video_fps
        if t > t_end:
            logger.debug(f'Reached {t_end=}')
            break

    cap.release()
    X = np.stack(feats)        # shape: [T, D]
    times = np.array(times)   # shape: [T]
    print(f'{X.shape=}, {times.shape=}')
    return X, times


X, times = extract_fft_features("./video/ekaterina.mp4", fps=2)

# %%
reducer = umap.UMAP(
    n_neighbors=10,
    min_dist=0.05,
    n_components=2,
    metric="euclidean",
    # random_state=42
)

Y = reducer.fit_transform(X)   # [T, 2]

plt.figure(figsize=(8, 6))
sc = plt.scatter(
    Y[:, 0], Y[:, 1],
    c=times,
    cmap="viridis",
    s=12
)
plt.plot(Y[:, 0], Y[:, 1], alpha=0.1)

plt.colorbar(sc, label="Time (s)")
plt.title("UMAP Manifold of Movie (FFT features)")
plt.axis("equal")
plt.tight_layout()
plt.show()

# %%

dmap = dm.DiffusionMap.from_sklearn(
    n_evecs=3,
    alpha=0.5,
    epsilon="bgh",
    k=64
)

phi = dmap.fit_transform(X)   # [T, n_evecs]

plt.figure(figsize=(8, 6))
sc = plt.scatter(
    phi[:, 1], phi[:, 2],      # 通常不用第 0 个特征
    c=times,
    cmap="plasma",
    s=12
)
plt.plot(phi[:, 1], phi[:, 2], alpha=0.1)

plt.colorbar(sc, label="Time (s)")
plt.title("Diffusion Map of Movie Dynamics")
plt.axis("equal")
plt.tight_layout()
plt.show()

# %%
