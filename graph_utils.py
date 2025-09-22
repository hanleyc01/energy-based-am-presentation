import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

pxw, pxh = 28, 28


def mask(x, pct_mask=0.3, seed: int = 11):
    rng = np.random.default_rng(seed=seed)
    prange = np.array([pct_mask, 1 - pct_mask])
    return x * rng.choice(np.asarray([-1.0, 1.0]), p=prange, size=x.shape)


def show_im(im: np.ndarray, title: str = "") -> None:
    im = im * 255.0
    im = rearrange(im, "(w h) 1 -> w h", w=pxw, h=pxh)
    plt.imshow(im)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def show_weights(W, title=""):
    plt.imshow(W, cmap="viridis", aspect="auto")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def display_pair(img1, img2, titles=["", ""]):
    images = [np.asarray(img).reshape(28, 28) for img in [img1, img2]]
    fig, axes = plt.subplots(1, 2, figsize=(5, 5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()


def display_recall(
    img1,
    img2,
    img3,
    titles=[
        "",
        "",
        "",
    ],
):
    images = [
        np.array(img).reshape(28, 28)
        for img in [
            img1,
            img2,
            img3,
        ]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
