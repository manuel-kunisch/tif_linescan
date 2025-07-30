import tifffile as tiff
from matplotlib.colors import to_rgb
from tifffile import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

# ------------------------------------
# 1. I/O
# ------------------------------------
def read_tif(path):
    """Returns ndarray with shape (Z, Y, X)."""
    im = imread(path)

    return im

# ------------------------------------
# 2. Normalisation helper
# ------------------------------------
def stretch(c, pct=(1, 99)):
    """Contrast-stretch a single channel to 0-1 floats."""
    lo, hi = np.percentile(c, pct)
    return rescale_intensity(c, in_range=(lo, hi), out_range=(0, 1)).astype(np.float32)

# ------------------------------------
# 3. Composite plotter
# ------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity   # pip install scikit-image

# ---------------------------------------------------------------------
def stretch(img, pct=(1, 99)):
    """Contrast-stretch each channel independently to 0-1."""
    low, high = np.percentile(img, pct)
    return rescale_intensity(img, in_range=(low, high),
                             out_range=(0, 1)).astype(np.float32)

def composite_channels(stack,
                       cmaps=("Reds", "Greens", "Blues", "magma", "cividis"),
                       pct=(1, 99),                 # per-channel contrast
                       figsize=(6, 6)):
    """
    stack   ndarray shaped (C, Y, X)
    cmaps   list/tuple with one entry per channel:
            • a Matplotlib colormap name **OR**
            • any valid colour string ("red", "#ff00ff", …)
    pct     stretch percentiles (vmin, vmax) for every channel
    """

    if stack.ndim != 3:
        raise ValueError("Expected (C, Y, X) array")

    C, Y, X = stack.shape
    if len(cmaps) < C:
        raise ValueError(f"{C} channels but only {len(cmaps)} colourmaps given")

    rgb = np.zeros((Y, X, 3), np.float32)

    for c in range(C):
        chan = stretch(stack[c], pct)            # 0-1 greyscale

        # --- turn greyscale → RGB by colourmap or solid colour --------------
        cm = cmaps[c]
        if cm in plt.colormaps:                  # a real colourmap name
            rgba = plt.get_cmap(cm)(chan)        # (Y,X,4)
            coloured = rgba[..., :3]             # drop alpha
        else:                                    # assume solid colour
            coloured = chan[..., None] * to_rgb(cm)

        # additive blend & clip to avoid overflow
        rgb = np.clip(rgb + coloured, 0, 1)

    # --- show ---------------------------------------------------------------
    plt.figure(figsize=figsize)
    plt.imshow(rgb, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return rgb              # in case you want to save it later

def stretch_fixed(chan, clim):
    return np.clip((chan - clim[0]) / (clim[1] - clim[0]), 0, 1)

def main():
    path = "data/composite_ao_flat_green_noao_red_linescan.tif"
    img = read_tif(path)
    composite_channels(img, pct=(1, 5))

if __name__ == "__main__":
    main()
