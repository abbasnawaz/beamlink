import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from beamlink.config import BeamMethod
from beamlink.data import Dataset, SceneMeta

logger = logging.getLogger("beamlink.offline")

METHOD_COLORS = {
    BeamMethod.SWEEP: "#2a78d6",
    BeamMethod.AI_TOP_K: "#eb6834",
    BeamMethod.MRT_WIDEBAND: "#1baf7a",
    BeamMethod.MRT_SUBCARRIER: "#eda100",
}
METHOD_LABELS = {
    BeamMethod.SWEEP: "DFT sweep",
    BeamMethod.AI_TOP_K: "AI top-K",
    BeamMethod.MRT_WIDEBAND: "Wideband MRT",
    BeamMethod.MRT_SUBCARRIER: "Per-subcarrier MRT",
}
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
NEUTRAL = "#52514e"
GRID_COLOR = "#e4e3df"
SAVE_DOTS_PER_INCH = 130
MAP_MARKER_SIZE = 4


def style_axes(panel: Axes) -> None:
    panel.grid(True, which="both", color=GRID_COLOR, linewidth=0.6)
    for side in ("top", "right"):
        panel.spines[side].set_visible(False)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=SAVE_DOTS_PER_INCH, bbox_inches="tight")
    plt.close(figure)
    logger.info(f"wrote {path} size_kb={path.stat().st_size / 1024:.0f}")


def positive(values: list[float]) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return np.where(array > 0, array, np.nan)


def map_panel(panel: Axes, meta: SceneMeta, image: npt.NDArray[np.float32], title: str) -> None:
    panel.imshow(image, extent=meta.image_extent_m, zorder=0)
    panel.plot(*meta.bs_position_m[:2], "^", color="black", ms=11, mfc="white", mew=2, zorder=3)
    panel.set_xlim(meta.grid_extent_m[0] - 2, meta.grid_extent_m[1] + 2)
    panel.set_ylim(meta.grid_extent_m[2] - 2, meta.grid_extent_m[3] + 2)
    panel.set_aspect("equal")
    panel.set_xlabel("x (m)")
    panel.set_ylabel("y (m)")
    panel.set_title(title)


def map_scatter(panel: Axes, dataset: Dataset, values: npt.NDArray[np.generic], **style: object) -> object:
    positions = dataset.positions_m
    return panel.scatter(
        positions[:, 0], positions[:, 1], c=values, s=MAP_MARKER_SIZE, marker="s", lw=0, **style
    )
