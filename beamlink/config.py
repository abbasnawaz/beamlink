import dataclasses
import math
import types
import typing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from beamlink.exceptions import ConfigError

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
THERMAL_NOISE_DBM_PER_HZ = -174.0
DBM_PER_DBW = 30.0

MAX_PLOTTED_SYMBOLS = 2000

DATASET_FILENAME = "dataset.npz"
META_FILENAME = "meta.json"
SCENE_IMAGE_FILENAME = "scene_topdown.png"
DATASET_SCHEMA_VERSION = 1

TOPDOWN_HALF_WIDTH_M = 100.0
TOPDOWN_RESOLUTION_PX = (1200, 780)

SIONNA_CFR_TOLERANCE = 1e-2

N_WIDE_BEAMS = 4
SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_TEST = 2
CHECKPOINT_FILENAME = "beam_predictor.pt"
PREDICTOR_FILENAME = "beam_predictor.onnx"
NORM_FILENAME = "norm.json"
NORM_SCHEMA_VERSION = 1
ONNX_INPUT_NAME = "features"
ONNX_OUTPUT_NAME = "logits"
ONNX_OPSET = 17
ONNX_PARITY_RELATIVE_TOLERANCE = 1e-5
ONNX_PARITY_SAMPLES = 1000
N_REFERENCE_SAMPLES = 16

APP_ANTENNA_OPTIONS = (4, 8, 16, 32)
APP_TX_POWER_RANGE_DBM = (10.0, 40.0)
APP_SNR_RANGE_DB = (-10.0, 30.0)
APP_DEFAULT_SNR_DB = 10.0
APP_PREFIX_RANGE = (0, 64)
APP_BER_SNR_DB = (-10.0, -6.0, -2.0, 2.0, 6.0, 10.0, 14.0, 18.0)
APP_BER_SLOTS = 3
APP_LINK_SLOTS = 1
APP_RAY_COUNT = 5
APP_DEFAULT_SEED = 0
APP_DEFAULT_POSITION_XY_M = (30.0, 0.0)

SYNTHETIC_N_PATHS = 8
SYNTHETIC_MAX_DELAY_S = 300e-9
SYNTHETIC_DELAY_DECAY_S = 100e-9
SYNTHETIC_AZIMUTH_SPAN = (-math.pi / 3.0, math.pi / 3.0)
SYNTHETIC_ZENITH_SPREAD = 0.1


class Modulation(StrEnum):
    QPSK = "qpsk"
    QAM16 = "16qam"
    QAM64 = "64qam"


class BeamMethod(StrEnum):
    SWEEP = "sweep"
    AI_TOP_K = "ai_topk"
    MRT_WIDEBAND = "mrt_wideband"
    MRT_SUBCARRIER = "mrt_subcarrier"


class CsiMode(StrEnum):
    PERFECT = "perfect"
    LS = "ls"


class SimulationMode(StrEnum):
    FREQUENCY = "freq"
    TIME = "time"


class FeatureSet(StrEnum):
    POSITION = "position"
    POSITION_WIDE_BEAMS = "position_wide_beams"


class Equalizer(StrEnum):
    ZF = "zf"
    MMSE = "mmse"


BITS_PER_SYMBOL = {Modulation.QPSK: 2, Modulation.QAM16: 4, Modulation.QAM64: 6}


@dataclass(frozen=True, slots=True)
class SystemConfig:
    carrier_frequency_hz: float
    subcarrier_spacing_hz: float
    n_subcarriers: int
    n_ofdm_symbols: int
    cyclic_prefix_length: int
    pilot_symbol_indices: tuple[int, ...]
    pilot_spacing: int
    pilot_offset: int
    pilot_seed: int
    n_antennas: int
    antenna_spacing_wavelengths: float
    bs_boresight_azimuth: float
    codebook_oversampling: int
    codebook_sector: tuple[float, ...] | None
    n_cir_precursor_taps: int
    n_cir_postcursor_taps: int
    equalizer: Equalizer
    tx_power_dbm: float
    noise_figure_db: float
    scene_name: str
    bs_position_xy_m: tuple[float, ...]
    bs_height_m: float
    ue_height_m: float
    ue_grid_extent_m: tuple[float, ...]
    grid_spacing_m: float
    building_clearance_m: float
    max_paths: int
    max_reflection_depth: int
    rt_samples_per_source: int
    rt_diffraction: bool
    rt_receivers_per_batch: int
    label_antenna_counts: tuple[int, ...]
    split_block_m: float
    split_fractions: tuple[float, ...]
    predictor_feature_set: FeatureSet
    predictor_hidden_units: int
    predictor_hidden_layers: int
    predictor_epochs: int
    predictor_batch_size: int
    predictor_learning_rate: float
    soft_label_temperature_db: float
    predictor_top_k: int

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT_M_PER_S / self.carrier_frequency_hz

    @property
    def sample_rate_hz(self) -> float:
        return self.n_subcarriers * self.subcarrier_spacing_hz

    @property
    def bandwidth_hz(self) -> float:
        return self.n_subcarriers * self.subcarrier_spacing_hz


def _validated_value(name: str, expected: typing.Any, value: object) -> object:
    if typing.get_origin(expected) is types.UnionType:
        options = typing.get_args(expected)
        if value is None and type(None) in options:
            return None
        expected = next(option for option in options if option is not type(None))
    if typing.get_origin(expected) is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{name}: expected a list, got {type(value).__name__}")
        element_type = typing.get_args(expected)[0]
        return tuple(_validated_value(name, element_type, element) for element in value)
    if isinstance(expected, type) and issubclass(expected, StrEnum):
        allowed = [member.value for member in expected]
        if value not in allowed:
            raise ConfigError(f"{name}: {value!r} is not one of {allowed}")
        return expected(value)
    if expected is bool and isinstance(value, bool):
        return value
    if expected is str and isinstance(value, str):
        return value
    if expected is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if expected is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ConfigError(f"{name}: expected {expected.__name__}, got {type(value).__name__} {value!r}")


def load_config(path: Path) -> SystemConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    hints = typing.get_type_hints(SystemConfig)
    expected_names = {field.name for field in dataclasses.fields(SystemConfig)}
    missing = sorted(expected_names - raw.keys())
    unknown = sorted(raw.keys() - expected_names)
    if missing or unknown:
        raise ConfigError(f"{path}: missing fields {missing}, unknown fields {unknown}")
    values = {name: _validated_value(name, hints[name], raw[name]) for name in expected_names}
    return SystemConfig(**values)
