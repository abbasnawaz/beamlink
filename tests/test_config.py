import math
from pathlib import Path

import pytest
import yaml

from beamlink.config import SPEED_OF_LIGHT_M_PER_S, Equalizer, load_config
from beamlink.exceptions import ConfigError

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _write_variant(tmp_path: Path, **overrides: object) -> Path:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw.update(overrides)
    variant = tmp_path / "variant.yaml"
    variant.write_text(yaml.safe_dump(raw))
    return variant


def test_default_config_loads_with_typed_fields() -> None:
    config = load_config(CONFIG_PATH)
    assert config.carrier_frequency_hz == pytest.approx(28e9)
    assert isinstance(config.n_subcarriers, int)
    assert config.pilot_symbol_indices == (0,)
    assert config.codebook_sector is None
    assert config.equalizer is Equalizer.MMSE
    assert config.sample_rate_hz == pytest.approx(256 * 120e3)
    assert config.wavelength_m == pytest.approx(SPEED_OF_LIGHT_M_PER_S / 28e9)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(_write_variant(tmp_path, fft_sise=128))


def test_missing_field_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    del raw["n_subcarriers"]
    variant = tmp_path / "variant.yaml"
    variant.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_config(variant)


def test_wrong_type_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(_write_variant(tmp_path, n_subcarriers="256"))
    with pytest.raises(ConfigError):
        load_config(_write_variant(tmp_path, equalizer="viterbi"))


def test_sector_list_becomes_float_tuple(tmp_path: Path) -> None:
    config = load_config(_write_variant(tmp_path, codebook_sector=[-1, 1]))
    assert config.codebook_sector == (-1.0, 1.0)


def test_default_config_has_four_tilted_sectors() -> None:
    config = load_config(CONFIG_PATH)
    assert config.sector_boresight_azimuths == pytest.approx((0.0, math.pi / 2, math.pi, 3 * math.pi / 2))
    assert config.sector_downtilt == pytest.approx(math.radians(10.0))
    assert config.element_beamwidth == pytest.approx(math.radians(65.0))
    assert config.n_sectors == 4
    assert config.n_beams == config.n_sectors * config.codebook_oversampling * config.n_antennas
