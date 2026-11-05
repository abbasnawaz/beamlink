import base64

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plots import CATEGORICAL, LOS_COLOR, METHOD_COLORS, METHOD_LABELS, NEUTRAL, NLOS_COLOR, PLOT_MARGIN

from beamlink.config import BeamMethod

LOBE_LENGTH_M = 45.0
RAY_LENGTH_M = 35.0
MAP_MARKER_SIZE = 5
NO_SERVICE_COLOR = "#8c8b85"


def _image_uri(image_png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")


def _scene_layout(
    figure: go.Figure, image_png: bytes, image_extent_m: tuple[float, ...], grid_extent_m: tuple[float, ...]
) -> None:
    x_min, x_max, y_min, y_max = image_extent_m
    figure.add_layout_image(
        source=_image_uri(image_png),
        xref="x",
        yref="y",
        x=x_min,
        y=y_max,
        sizex=x_max - x_min,
        sizey=y_max - y_min,
        sizing="stretch",
        layer="below",
    )
    figure.update_xaxes(
        range=[grid_extent_m[0] - 3, grid_extent_m[1] + 3], title="x (m)", showgrid=False, constrain="domain"
    )
    figure.update_yaxes(
        range=[grid_extent_m[2] - 3, grid_extent_m[3] + 3],
        title="y (m)",
        showgrid=False,
        scaleanchor="x",
        constrain="domain",
    )


def _bs_marker(bs_position_m: tuple[float, ...]) -> go.Scattergl:
    return go.Scattergl(
        x=[bs_position_m[0]],
        y=[bs_position_m[1]],
        mode="markers",
        marker={
            "symbol": "triangle-up",
            "size": 16,
            "color": "white",
            "line": {"color": "black", "width": 2},
        },
        name="Base station",
        hoverinfo="name",
    )


def build_scene_map_figure(
    *,
    image_png: bytes,
    image_extent_m: tuple[float, ...],
    grid_extent_m: tuple[float, ...],
    positions_m: npt.NDArray[np.float32],
    is_los: npt.NDArray[np.bool_],
    bs_position_m: tuple[float, ...],
    selected_idx: int,
    lobe_azimuth: npt.NDArray[np.float32],
    lobe_gain: npt.NDArray[np.float32],
    ray_azimuth: npt.NDArray[np.float32],
    ray_power_db: npt.NDArray[np.float32],
    method: BeamMethod,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=positions_m[:, 0],
            y=positions_m[:, 1],
            mode="markers",
            marker={
                "size": MAP_MARKER_SIZE,
                "color": np.where(is_los, LOS_COLOR, NLOS_COLOR),
                "opacity": 0.45,
            },
            customdata=np.arange(positions_m.shape[0]),
            hovertemplate="UE #%{customdata}<br>x = %{x:.1f} m<br>y = %{y:.1f} m<extra></extra>",
            name="UE grid (blue LOS, orange NLOS)",
        )
    )
    lobe_x = bs_position_m[0] + LOBE_LENGTH_M * lobe_gain * np.cos(lobe_azimuth)
    lobe_y = bs_position_m[1] + LOBE_LENGTH_M * lobe_gain * np.sin(lobe_azimuth)
    figure.add_trace(
        go.Scatter(
            x=np.concatenate([[bs_position_m[0]], lobe_x, [bs_position_m[0]]]),
            y=np.concatenate([[bs_position_m[1]], lobe_y, [bs_position_m[1]]]),
            mode="lines",
            fill="toself",
            line={"color": METHOD_COLORS[method], "width": 2},
            opacity=0.55,
            name=f"Beam pattern ({METHOD_LABELS[method]})",
            hoverinfo="name",
        )
    )
    strongest = float(np.max(ray_power_db)) if ray_power_db.size else 0.0
    for rank, (azimuth, power_db) in enumerate(zip(ray_azimuth, ray_power_db, strict=True)):
        figure.add_trace(
            go.Scatter(
                x=[bs_position_m[0], bs_position_m[0] + RAY_LENGTH_M * np.cos(azimuth)],
                y=[bs_position_m[1], bs_position_m[1] + RAY_LENGTH_M * np.sin(azimuth)],
                mode="lines",
                line={"color": NEUTRAL, "width": max(1.0, 4.0 + (power_db - strongest) / 5.0), "dash": "dot"},
                name="Strongest departure directions" if rank == 0 else None,
                showlegend=rank == 0,
                hovertemplate=f"path {rank + 1}: {power_db:.1f} dB<extra></extra>",
            )
        )
    selected = positions_m[selected_idx]
    figure.add_trace(
        go.Scatter(
            x=[selected[0]],
            y=[selected[1]],
            mode="markers",
            marker={"symbol": "star", "size": 18, "color": "#e34948", "line": {"color": "black", "width": 1}},
            name=f"Selected UE #{selected_idx}",
            hoverinfo="name",
        )
    )
    figure.add_trace(_bs_marker(bs_position_m))
    _scene_layout(figure, image_png, image_extent_m, grid_extent_m)
    figure.update_layout(
        title="Street canyon: click a UE position",
        height=640,
        margin=PLOT_MARGIN,
        legend={"orientation": "h", "y": -0.15},
        clickmode="event+select",
        dragmode="pan",
    )
    return figure


def build_coverage_figure(
    *,
    image_png: bytes,
    image_extent_m: tuple[float, ...],
    grid_extent_m: tuple[float, ...],
    positions_m: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    bs_position_m: tuple[float, ...],
    selected_idx: int,
    title: str,
    colorbar_title: str,
    colorscale: str | list[list[object]],
    value_range: tuple[float, float],
    no_service: npt.NDArray[np.bool_] | None = None,
    no_service_label: str = "",
) -> go.Figure:
    position_idx = np.arange(positions_m.shape[0])
    served = np.ones(positions_m.shape[0], dtype=np.bool_) if no_service is None else ~no_service
    figure = go.Figure(
        go.Scattergl(
            x=positions_m[served, 0],
            y=positions_m[served, 1],
            mode="markers",
            marker={
                "size": MAP_MARKER_SIZE + 1,
                "symbol": "square",
                "color": values[served],
                "colorscale": colorscale,
                "cmin": value_range[0],
                "cmax": value_range[1],
                "colorbar": {"title": colorbar_title},
            },
            customdata=np.stack([position_idx[served], values[served]], axis=-1),
            hovertemplate="UE #%{customdata[0]}<br>%{customdata[1]:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    if no_service is not None:
        figure.add_trace(
            go.Scattergl(
                x=positions_m[no_service, 0],
                y=positions_m[no_service, 1],
                mode="markers",
                marker={"size": MAP_MARKER_SIZE + 1, "symbol": "square", "color": NO_SERVICE_COLOR},
                customdata=np.stack([position_idx[no_service], values[no_service]], axis=-1),
                hovertemplate="UE #%{customdata[0]}<br>%{customdata[1]:.2f} (no service)<extra></extra>",
                name=no_service_label,
            )
        )
    selected = positions_m[selected_idx]
    figure.add_trace(
        go.Scattergl(
            x=[selected[0]],
            y=[selected[1]],
            mode="markers",
            marker={"symbol": "star", "size": 16, "color": "#e34948", "line": {"color": "black", "width": 1}},
            name=f"Selected UE #{selected_idx}",
        )
    )
    figure.add_trace(_bs_marker(bs_position_m))
    _scene_layout(figure, image_png, image_extent_m, grid_extent_m)
    figure.update_layout(title=title, height=560, margin=PLOT_MARGIN, legend={"orientation": "h", "y": -0.12})
    return figure


def categorical_colorscale(n_categories: int) -> list[list[object]]:
    colors = [CATEGORICAL[idx % len(CATEGORICAL)] for idx in range(n_categories)]
    edges = np.linspace(0.0, 1.0, n_categories + 1)
    scale: list[list[object]] = []
    for idx, color in enumerate(colors):
        scale += [[float(edges[idx]), color], [float(edges[idx + 1]), color]]
    return scale
