import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from beamlink.config import BeamMethod

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
LOS_COLOR = "#2a78d6"
NLOS_COLOR = "#eb6834"
NEUTRAL = "#52514e"
FAINT = "#c3c2b7"
PLOT_MARGIN = {"l": 50, "r": 20, "t": 50, "b": 45}
PATTERN_FLOOR_DB = -30.0
POLAR_BORESIGHT_DEG = 90.0
POLAR_TICKS_DEG = np.arange(-90.0, 91.0, 30.0)


def build_beam_pattern_figure(
    relative_azimuth_deg: npt.NDArray[np.float32],
    pattern_db: npt.NDArray[np.float32],
    codebook_pattern_db: npt.NDArray[np.float32] | None,
    method: BeamMethod,
) -> go.Figure:
    figure = go.Figure()
    if codebook_pattern_db is not None:
        for beam_idx, beam_db in enumerate(codebook_pattern_db):
            figure.add_trace(
                go.Scatterpolar(
                    r=np.maximum(beam_db, PATTERN_FLOOR_DB),
                    theta=relative_azimuth_deg + POLAR_BORESIGHT_DEG,
                    mode="lines",
                    line={"color": FAINT, "width": 1},
                    name="Codebook beams" if beam_idx == 0 else None,
                    showlegend=beam_idx == 0,
                    hoverinfo="skip",
                )
            )
    figure.add_trace(
        go.Scatterpolar(
            r=np.maximum(pattern_db, PATTERN_FLOOR_DB),
            theta=relative_azimuth_deg + POLAR_BORESIGHT_DEG,
            mode="lines",
            line={"color": METHOD_COLORS[method], "width": 3},
            name=METHOD_LABELS[method],
            hovertemplate="%{theta:.0f} deg: %{r:.1f} dBi<extra></extra>",
        )
    )
    figure.update_layout(
        title="Beam pattern (gain vs angle from boresight, dBi)",
        height=420,
        margin=PLOT_MARGIN,
        polar={
            "sector": [0, 180],
            "angularaxis": {
                "direction": "counterclockwise",
                "tickvals": POLAR_TICKS_DEG + POLAR_BORESIGHT_DEG,
                "ticktext": [f"{tick:+.0f} deg" for tick in POLAR_TICKS_DEG],
            },
            "radialaxis": {"range": [PATTERN_FLOOR_DB, float(np.max(pattern_db)) + 2], "angle": 0},
        },
        legend={"orientation": "h", "y": -0.05},
    )
    return figure


def build_pdp_figure(delay_ns: npt.NDArray[np.float32], power_db: npt.NDArray[np.float32]) -> go.Figure:
    floor = float(np.min(power_db)) - 5.0 if power_db.size else 0.0
    stem_x = np.repeat(delay_ns, 3).astype(object)
    stem_y = np.stack(
        [np.full_like(power_db, floor), power_db, np.full_like(power_db, np.nan)], axis=-1
    ).ravel()
    stem_x[2::3] = None
    figure = go.Figure(
        [
            go.Scatter(
                x=stem_x, y=stem_y, mode="lines", line={"color": LOS_COLOR, "width": 2}, hoverinfo="skip"
            ),
            go.Scatter(
                x=delay_ns,
                y=power_db,
                mode="markers",
                marker={"color": LOS_COLOR, "size": 8},
                hovertemplate="%{x:.0f} ns: %{y:.1f} dB<extra></extra>",
            ),
        ]
    )
    figure.update_layout(
        title="Power delay profile",
        xaxis_title="Delay after first arrival (ns)",
        yaxis_title="Path gain (dB)",
        showlegend=False,
        height=360,
        margin=PLOT_MARGIN,
    )
    return figure


def build_channel_response_figure(
    frequency_megahertz: npt.NDArray[np.float32],
    single_antenna_db: npt.NDArray[np.float32],
    effective_db: npt.NDArray[np.float32],
    estimated_db: npt.NDArray[np.float32],
    method: BeamMethod,
    unit_label: str,
) -> go.Figure:
    figure = go.Figure(
        [
            go.Scatter(
                x=frequency_megahertz, y=single_antenna_db, name="Antenna 0 alone", line={"color": NEUTRAL}
            ),
            go.Scatter(
                x=frequency_megahertz,
                y=effective_db,
                name=f"Beamformed h_eff ({METHOD_LABELS[method]})",
                line={"color": METHOD_COLORS[method], "width": 3},
            ),
            go.Scatter(
                x=frequency_megahertz,
                y=estimated_db,
                name="Receiver's estimate of h_eff",
                line={"color": METHOD_COLORS[method], "width": 1.5, "dash": "dot"},
            ),
        ]
    )
    figure.update_layout(
        title="Channel across subcarriers",
        xaxis_title="Baseband frequency (MHz)",
        yaxis_title=f"|H| (dB, {unit_label})",
        height=360,
        margin=PLOT_MARGIN,
        legend={"orientation": "h", "y": -0.25},
    )
    return figure


def build_constellation_figure(
    before: npt.NDArray[np.complex64], after: npt.NDArray[np.complex64], ideal: npt.NDArray[np.complex64]
) -> go.Figure:
    scale = float(np.max(np.abs(ideal))) * 1.6
    figure = go.Figure(
        [
            go.Scatter(
                x=ideal.real,
                y=ideal.imag,
                mode="markers",
                marker={"symbol": "cross-thin", "size": 14, "line": {"width": 1.5, "color": FAINT}},
                name="Ideal points",
            ),
            go.Scatter(
                x=after.real,
                y=after.imag,
                mode="markers",
                marker={"size": 4, "color": LOS_COLOR, "opacity": 0.6},
                name="After equalization",
            ),
        ]
    )
    before_scale = float(np.percentile(np.abs(before), 95)) if before.size else 1.0
    normalized_before = before / max(before_scale, np.finfo(np.float32).tiny) * float(np.max(np.abs(ideal)))
    figure.add_trace(
        go.Scatter(
            x=normalized_before.real,
            y=normalized_before.imag,
            mode="markers",
            marker={"size": 3, "color": NLOS_COLOR, "opacity": 0.35},
            name="Received, before equalization (rescaled)",
            visible="legendonly",
        )
    )
    figure.update_layout(
        title="Constellation",
        xaxis={"title": "In-phase", "range": [-scale, scale], "constrain": "domain"},
        yaxis={"title": "Quadrature", "range": [-scale, scale], "scaleanchor": "x", "constrain": "domain"},
        height=420,
        margin=PLOT_MARGIN,
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def build_ber_figure(
    snr_db: npt.NDArray[np.float32],
    ber_by_method: dict[BeamMethod, npt.NDArray[np.float32]],
    theory_snr_db: npt.NDArray[np.float32],
    theory_ber: npt.NDArray[np.float32],
    title: str,
) -> go.Figure:
    figure = go.Figure()
    markers = {
        BeamMethod.SWEEP: "circle-open",
        BeamMethod.AI_TOP_K: "square-open",
        BeamMethod.MRT_WIDEBAND: "triangle-up-open",
        BeamMethod.MRT_SUBCARRIER: "diamond-open",
    }
    for method, ber in ber_by_method.items():
        figure.add_trace(
            go.Scatter(
                x=snr_db,
                y=np.where(ber > 0, ber, np.nan),
                mode="lines+markers",
                marker={"symbol": markers[method], "size": 10},
                line={"color": METHOD_COLORS[method], "width": 2},
                name=METHOD_LABELS[method],
            )
        )
    figure.add_trace(
        go.Scatter(
            x=theory_snr_db,
            y=theory_ber,
            mode="lines",
            line={"color": NEUTRAL, "dash": "dash"},
            name="AWGN theory (no beamforming gain)",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Per-antenna SNR before beamforming (dB)",
        yaxis={"title": "Bit error rate", "type": "log", "range": [-5, 0]},
        height=480,
        margin=PLOT_MARGIN,
    )
    return figure
