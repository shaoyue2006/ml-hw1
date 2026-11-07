"""Render classroom figures from the fitted MLP and the published test group table."""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PRES = Path(__file__).resolve().parents[1]
DEST = PRES / "figures/04_mlp"
INK, MUTED = "#152C42", "#465D6B"
BLUE, TEAL = "#345D88", "#117B72"


def canvas():
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 16), ylim=(0, 9))
    ax.axis("off")
    return fig, ax


def save(fig, name):
    DEST.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(DEST / f"{name}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def architecture():
    import tomli
    from torch import nn

    v3 = PRES.parent / "v3"
    sys.path.insert(0, str(v3 / "src"))
    from experiment import restore

    m, (pre, target) = restore(v3 / "results/final_model")
    with (v3 / "configs/mlp.toml").open("rb") as fh:
        cfg = tomli.load(fh)
    raw = pre.n_features_in_
    numeric = len(pre.transformers_[0][2])
    categories = pre.named_transformers_["categorical"]["onehot"].categories_
    encoded = sum(map(len, categories))
    width = len(pre.get_feature_names_out())
    assert (
        raw == numeric + len(categories) and width == numeric + encoded == m.layers[0].in_features
    )
    assert [type(layer) for layer in m.layers] == [
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
    ]
    assert [layer.out_features for layer in m.layers if isinstance(layer, nn.Linear)] == [
        *cfg["hidden"],
        1,
    ]
    assert target.n_features_in_ == 1
    fig, ax = canvas()

    def label(x, y, s, size, **kwargs):
        ax.text(x, y, s, fontsize=size, va="center", color=kwargs.pop("color", INK), **kwargs)

    label(0.60, 8.20, "共享单车需求预测的 MLP 回归网络", 30, weight="bold")
    label(0.62, 7.57, "Regression   /   完整开发集重训模型", 14, color=MUTED)
    ax.axhline(7.13, xmin=0.037, xmax=0.963, color="#C8D6DB", lw=1)
    label(0.62, 6.54, f"原始输入  ·  {raw} 个字段", 18, weight="bold")
    label(0.62, 5.96, f"{numeric} 个连续变量：中位数填充、标准化", 13, color=MUTED)
    label(0.62, 5.48, f"{len(categories)} 个类别变量：众数填充、独热编码", 13, color=MUTED)
    ax.annotate(
        "",
        xy=(8.32, 6.12),
        xytext=(7.18, 6.12),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color=TEAL),
    )
    label(8.60, 6.54, f"网络输入  ·  {width} 维", 18, weight="bold", color=TEAL)
    label(8.60, 5.94, f"{numeric} 个连续分量  +  {encoded} 个独热分量", 14, color=MUTED)
    label(8.60, 5.48, "独热列数由当前训练窗口决定", 12, color=MUTED)
    # Use the loaded PyTorch module order and its fitted layer shapes.
    nodes = [("Input", f"{width} 维", 1.60)]
    for layer in m.layers:
        nodes.append(
            ("Linear", f"({layer.in_features}, {layer.out_features})", 2.24)
            if isinstance(layer, nn.Linear)
            else ("ReLU", "激活", 1.37)
        )
    nodes.append(("Output", "1 个标量", 1.70))
    gap = 0.45
    x = (16 - sum(n[2] for n in nodes) - gap * (len(nodes) - 1)) / 2
    for i, (title, sub, w) in enumerate(nodes):
        activation = title == "ReLU"
        edge = TEAL if activation else BLUE
        ax.add_patch(
            FancyBboxPatch(
                (x, 3.46),
                w,
                1.22,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                facecolor="#E3F3F0" if activation else "#EAF1F8",
                edgecolor=edge,
                lw=1.2,
            )
        )
        label(x + w / 2, 4.25, title, 16, ha="center", weight="bold", color=edge)
        label(x + w / 2, 3.80, sub, 12, ha="center")
        x += w
        if i < len(nodes) - 1:
            ax.annotate(
                "",
                xy=(x + gap - 0.035, 4.07),
                xytext=(x + 0.035, 4.07),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=MUTED),
            )
            x += gap
    label(0.62, 2.89, "线性输出预测标准化目标，逆标准化后得到每小时租赁量。", 14, color=MUTED)
    ax.axhline(2.31, xmin=0.037, xmax=0.963, color="#C8D6DB", lw=1)
    label(0.62, 1.65, "MSE Loss", 18, weight="bold", color=BLUE)
    label(0.62, 0.96, "在标准化目标上计算均方误差", 13)
    label(8.60, 1.65, "Adam Optimizer", 18, weight="bold", color=TEAL)
    label(8.60, 0.96, f"学习率 {cfg['learning_rate']:g}    ·    批大小 {cfg['batch_size']}", 13)
    save(fig, "mlp_architecture")
    print(f"Architecture verified: {raw} raw, {numeric}+{encoded} encoded; {m.layers}")


def demand():
    tab = pd.read_csv(PRES / "tables/test_demand_groups.csv")
    order = {"≤37": "低需求", "38–258": "中需求", ">258": "高需求"}
    if (
        set(tab.split) != {"test"}
        or len(tab) != len(order)
        or set(tab.actual_rentals_per_hour) != set(order)
    ):
        raise ValueError("Unexpected group table; verify thresholds and split")
    tab = tab.set_index("actual_rentals_per_hour").loc[list(order)]
    values = tab[["tuned_tree_mae", "mlp_mae", "hours"]].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or not (tab.hours > 0).all():
        raise ValueError("Missing or invalid demand-group data")
    fig, header = canvas()
    header.text(
        0.65,
        8.18,
        "不同需求水平下的独立测试误差",
        fontsize=29,
        color=INK,
        weight="bold",
        va="center",
    )
    header.text(
        0.67,
        7.48,
        f"测试集共 {tab.hours.sum():,} 个小时   /   MAE 越低越好",
        fontsize=14,
        color=MUTED,
        va="center",
    )
    ax = fig.add_axes([0.115, 0.205, 0.84, 0.53])
    x = np.arange(len(tab))
    for offset, key, name, color in [
        (-0.18, "tuned_tree_mae", "v2 调参原生类别提升树", BLUE),
        (0.18, "mlp_mae", "v3 PyTorch MLP", TEAL),
    ]:
        bars = ax.bar(x + offset, tab[key], 0.33, color=color, label=name, zorder=3)
        ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=15, color=INK)
    ax.set_ylim(0, tab[["tuned_tree_mae", "mlp_mae"]].to_numpy().max() * 1.20)
    ax.set_xticks(
        x,
        [
            f"{name}（{threshold}）\n{int(tab.loc[threshold, 'hours']):,} 小时"
            for threshold, name in order.items()
        ],
        fontsize=13,
    )
    ax.set_xlabel("按真实每小时租赁量分组（辆次）", fontsize=13, labelpad=10)
    ax.set_ylabel("平均绝对误差 MAE（辆次/小时）", fontsize=13, labelpad=8)
    ax.grid(axis="y", color="#DFE7EB", lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBC9D0")
    ax.tick_params(length=0, pad=6, colors=INK)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.015), frameon=False, ncol=2, fontsize=12)
    save(fig, "demand_group_comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figure", choices=["architecture", "demand", "all"], default="all", nargs="?"
    )
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "DejaVu Sans"],
            "svg.fonttype": "path",
            "axes.unicode_minus": False,
        }
    )
    if args.figure in {"architecture", "all"}:
        architecture()
    if args.figure in {"demand", "all"}:
        demand()
