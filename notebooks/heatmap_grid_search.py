import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

gammas = [0.95, 0.99, 0.999]
lrs = ["1e-4", "3e-4", "1e-3"]

data = {
    "PPO on Hopper-v4": np.array([
        [395433615, 2107331790, 957257189],
        [403634731, 1713381980, 914941733],
        [464184675, 1330086223, 1808345064],
    ]),
    "PPO on HumanoidStandup-v4": np.array([
        [87814579602, 122425951707, 65930061916],
        [91556308375, 103690117707, 53222830086],
        [78846959535, 68630049727, 60633606523],
    ]),
    "SAC on Hopper-v4": np.array([
        [155108785, 741166484, 722000406],
        [113179810, 861489997, 374537775],
        [150320761, 878444278, 611304399],
    ]),
    "SAC on HumanoidStandup-v4": np.array([
        [52165743484, 54347068652, 51274429547],
        [47345793292, 53975889781, 49787879785],
        [53984983430, 51723384805, 46566391410],
    ]),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for ax, (title, values) in zip(axes, data.items()):
    # Normalize to [0, 1] for color mapping within each subplot
    vmin, vmax = values.min(), values.max()

    im = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(3):
        for j in range(3):
            val = values[i, j]
            if val >= 1e9:
                label = f"{val/1e9:.1f}B"
            elif val >= 1e6:
                label = f"{val/1e6:.0f}M"
            else:
                label = f"{val:.0f}"

            if val == vmax:
                label += " *"

            text_color = "white" if (val - vmin) / (vmax - vmin) > 0.6 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=11, fontweight="bold", color=text_color)

    ax.set_xticks(range(3))
    ax.set_xticklabels([f"$\\gamma$={g}" for g in gammas])
    ax.set_yticks(range(3))

    lr_prefix = "policy_lr" if "SAC" in title else "lr"
    ax.set_yticklabels([f"{lr_prefix}={lr}" for lr in lrs])

    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Grid Search AUC Heatmaps", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("grid_search_heatmaps.pdf", bbox_inches="tight")
plt.show()
