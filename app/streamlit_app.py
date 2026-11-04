from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import streamlit as st
from explore_tabs import COVERAGE_VIEWS, render_about_tab, render_ber_tab, render_coverage_tab
from link_tab import (
    MODULATION_LABELS,
    AppResources,
    Controls,
    ai_available,
    available_methods,
    render_link_tab,
)
from plots import METHOD_LABELS

from beamlink.config import (
    APP_ANTENNA_OPTIONS,
    APP_DEFAULT_SEED,
    APP_DEFAULT_SNR_DB,
    APP_PREFIX_RANGE,
    APP_SNR_RANGE_DB,
    APP_TX_POWER_RANGE_DBM,
    SPLIT_TEST,
    CsiMode,
    Modulation,
    SimulationMode,
    SystemConfig,
    load_config,
)
from beamlink.data import Dataset, SceneMeta, load_dataset, load_meta, load_scene_image, predictor_features
from beamlink.exceptions import ArtifactError, BeamLinkError
from beamlink.linkbudget import received_snr_db
from beamlink.metrics import best_candidate_power_db, top_k_accuracy, top_k_power_loss_db
from beamlink.predictor import BeamPredictor, load_predictor, predict_top_k

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DATA_DIR = REPO_ROOT / "data"
CSI_LABELS = {CsiMode.PERFECT: "Perfect CSI", CsiMode.LS: "LS pilot estimate"}
SIMULATION_LABELS = {
    SimulationMode.FREQUENCY: "Frequency domain (fast)",
    SimulationMode.TIME: "Time domain (real CP)",
}

st.set_page_config(page_title="BeamLink", page_icon="📡", layout="wide")


@st.cache_resource
def get_config() -> SystemConfig:
    return load_config(CONFIG_PATH)


@st.cache_resource
def get_dataset() -> Dataset:
    return load_dataset(DATA_DIR)


@st.cache_resource
def get_meta() -> SceneMeta:
    return load_meta(DATA_DIR)


@st.cache_resource
def get_scene_image() -> bytes:
    return load_scene_image(DATA_DIR)


@st.cache_resource
def get_predictor() -> BeamPredictor | None:
    try:
        return load_predictor(DATA_DIR)
    except ArtifactError:
        return None


@st.cache_resource
def get_ai_ranking(n_candidates: int) -> npt.NDArray[np.int64] | None:
    predictor = get_predictor()
    if predictor is None:
        return None
    features = predictor_features(get_dataset(), predictor.norm.feature_set)
    return predict_top_k(predictor, features, n_candidates=n_candidates)


@st.cache_resource
def get_coverage(n_antennas: int, tx_power_dbm: float, top_k: int) -> dict[str, npt.NDArray[np.float32]]:
    config, dataset = get_config(), get_dataset()
    power_db = dataset.labels.beam_power_db[n_antennas]
    budget = {
        "tx_power_dbm": tx_power_dbm,
        "bandwidth_hz": config.bandwidth_hz,
        "noise_figure_db": config.noise_figure_db,
    }
    coverage = {
        COVERAGE_VIEWS[0]: received_snr_db(power_db.max(axis=-1), **budget),
        COVERAGE_VIEWS[3]: dataset.labels.best_beam[n_antennas].astype(np.float32),
        COVERAGE_VIEWS[4]: dataset.is_los.astype(np.float32),
    }
    predictor, ranking = get_predictor(), get_ai_ranking(top_k)
    if predictor is not None and predictor.norm.n_antennas == n_antennas and ranking is not None:
        coverage[COVERAGE_VIEWS[1]] = received_snr_db(best_candidate_power_db(power_db, ranking), **budget)
        coverage[COVERAGE_VIEWS[2]] = top_k_power_loss_db(power_db, ranking)
    return coverage


@st.cache_resource
def get_model_card(top_k: int) -> dict[str, float]:
    predictor, dataset, ranking = get_predictor(), get_dataset(), get_ai_ranking(top_k)
    test = dataset.labels.split == SPLIT_TEST
    n_antennas = predictor.norm.n_antennas
    best = dataset.labels.best_beam[n_antennas][test].astype(np.int64)
    loss_db = top_k_power_loss_db(dataset.labels.beam_power_db[n_antennas][test], ranking[test])
    return {
        "top-1 accuracy": top_k_accuracy(ranking[test, :1], best),
        f"top-{top_k} accuracy": top_k_accuracy(ranking[test], best),
        f"mean power loss, K={top_k} (dB)": float(loss_db.mean()),
        f"95th-percentile power loss, K={top_k} (dB)": float(np.percentile(loss_db, 95)),
        "test positions": float(test.sum()),
    }


def get_resources() -> AppResources:
    return AppResources(
        config=get_config(),
        dataset=get_dataset(),
        meta=get_meta(),
        scene_image=get_scene_image(),
        predictor=get_predictor(),
        ai_ranking=get_ai_ranking,
        coverage=get_coverage,
        model_card=get_model_card,
    )


def sidebar_controls(resources: AppResources) -> Controls:
    base = resources.config
    st.sidebar.header("System")
    n_antennas = st.sidebar.select_slider("BS antennas N", options=APP_ANTENNA_OPTIONS, value=base.n_antennas)
    tx_power_dbm, snr_db = None, None
    if st.sidebar.toggle("Fixed SNR (normalised channel)", value=False):
        snr_db = st.sidebar.slider("Per-antenna SNR (dB)", *APP_SNR_RANGE_DB, APP_DEFAULT_SNR_DB, step=1.0)
    else:
        tx_power_dbm = st.sidebar.slider(
            "Tx power (dBm)", *APP_TX_POWER_RANGE_DBM, base.tx_power_dbm, step=1.0
        )
    modulation = st.sidebar.selectbox(
        "Modulation", list(Modulation), index=1, format_func=MODULATION_LABELS.get
    )
    st.sidebar.header("Beam selection")
    methods = available_methods(resources, n_antennas)
    method = st.sidebar.selectbox("Beam method", methods, format_func=METHOD_LABELS.get)
    if not ai_available(resources, n_antennas) and resources.predictor is not None:
        st.sidebar.caption(f"AI top-K needs N = {resources.predictor.norm.n_antennas}.")
    n_beams = n_antennas * base.codebook_oversampling
    top_k = st.sidebar.slider("AI candidates K (beams)", 1, n_beams, min(base.predictor_top_k, n_beams))
    st.sidebar.header("Receiver")
    csi_mode = st.sidebar.radio("Channel knowledge", list(CsiMode), index=1, format_func=CSI_LABELS.get)
    simulation_mode = st.sidebar.radio("Simulation", list(SimulationMode), format_func=SIMULATION_LABELS.get)
    prefix_length = base.cyclic_prefix_length
    if simulation_mode is SimulationMode.TIME:
        prefix_length = st.sidebar.slider(
            "Cyclic prefix (samples)", *APP_PREFIX_RANGE, base.cyclic_prefix_length
        )
    seed = int(st.sidebar.number_input("Random seed", min_value=0, value=APP_DEFAULT_SEED, step=1))
    config = replace(base, n_antennas=n_antennas, cyclic_prefix_length=prefix_length)
    return Controls(config, tx_power_dbm, snr_db, modulation, method, top_k, csi_mode, simulation_mode, seed)


def main() -> None:
    st.title("BeamLink: AI-assisted beam selection over a ray-traced MIMO-OFDM channel")
    try:
        resources = get_resources()
        controls = sidebar_controls(resources)
        link, coverage, ber, about = st.tabs(["Link simulator", "Coverage explorer", "BER curves", "About"])
        with link:
            render_link_tab(resources, controls)
        with coverage:
            render_coverage_tab(resources, controls)
        with ber:
            render_ber_tab(resources, controls)
        with about:
            render_about_tab(resources)
    except BeamLinkError as error:
        st.error(f"{type(error).__name__}: {error}")


main()
