from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from beamlink.array import beam_pattern, sectored_array_response
from beamlink.beamforming import (
    beamforming_gain,
    effective_channel,
    precode_grid,
    sectored_codebook,
    serving_sector,
    spread_over_subcarriers,
    strongest_beam_idx,
    subcarrier_mrt_weights,
    sweep_beams,
    wideband_mrt_weights,
    wideband_power,
)
from beamlink.channel import (
    cir_to_cfr,
    n_cir_taps,
    paths_to_cfr,
    paths_to_cir,
    power_delay_profile,
    time_aligned_delays,
    unit_power_gain,
)
from beamlink.config import (
    MAX_PLOTTED_SYMBOLS,
    BeamMethod,
    CsiMode,
    Equalizer,
    Modulation,
    SimulationMode,
    SystemConfig,
)
from beamlink.data import ChannelPaths
from beamlink.equalization import mmse_equalize, zf_equalize
from beamlink.estimation import interpolate_pilots, linear_interpolation_matrix, ls_estimate
from beamlink.exceptions import ConfigError, ShapeError
from beamlink.linkbudget import linear_to_db, noise_variance_for_snr, noise_variance_for_tx_power
from beamlink.metrics import bit_error_rate, effective_snr, power_loss_db, spectral_efficiency
from beamlink.modulation import bits_per_symbol, demap_hard, map_bits
from beamlink.ofdm import (
    apply_frequency_channel,
    apply_time_channel,
    extract_data_symbols,
    map_resource_grid,
    ofdm_demodulate,
    ofdm_modulate,
    pilot_mask,
    pilot_subcarrier_idx,
    pilot_values,
)

ComplexArray = npt.NDArray[np.complex64]
FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class LinkResult:
    method: BeamMethod
    beam_idx: int | None
    serving_sector: int
    beams_tested: int
    beam_weights: ComplexArray
    snr_db: float
    beamforming_gain_db: float
    ber: float
    n_bits: int
    spectral_efficiency: float
    power_loss_db: float
    h_eff_true: ComplexArray
    h_eff_estimated: ComplexArray
    symbols_before_equalization: ComplexArray
    symbols_after_equalization: ComplexArray
    pdp_delay_s: FloatArray
    pdp: FloatArray
    cfr_reference_antenna: ComplexArray


@dataclass(frozen=True, slots=True)
class _Channel:
    path_gain: ComplexArray
    path_delay_s: FloatArray
    cfr: ComplexArray
    cir: ComplexArray | None
    cfr_seen_by_ofdm: ComplexArray
    isotropic_cfr: ComplexArray


@dataclass(frozen=True, slots=True)
class _BeamSelection:
    beam_weights: ComplexArray
    subcarrier_weights: ComplexArray
    beam_idx: int | None
    beams_tested: int


@dataclass(frozen=True, slots=True)
class _Transmission:
    bits: npt.NDArray[np.uint8]
    grid: ComplexArray
    mask: npt.NDArray[np.bool_]
    known_pilots: ComplexArray


def _transmission(
    config: SystemConfig, modulation: Modulation, n_slots: int, rng: np.random.Generator
) -> _Transmission:
    mask = pilot_mask(
        n_ofdm_symbols=config.n_ofdm_symbols,
        n_subcarriers=config.n_subcarriers,
        pilot_symbol_indices=config.pilot_symbol_indices,
        pilot_spacing=config.pilot_spacing,
        pilot_offset=config.pilot_offset,
    )
    known_pilots = pilot_values(mask, rng=np.random.default_rng(config.pilot_seed))
    n_bits = int(np.count_nonzero(~mask)) * bits_per_symbol(modulation)
    bits = rng.integers(0, 2, size=(n_slots, n_bits), dtype=np.uint8)
    grid = map_resource_grid(map_bits(bits, modulation=modulation), known_pilots, mask)
    return _Transmission(bits, grid, mask, known_pilots)


def _noise_variance(config: SystemConfig, snr_db: float | None, tx_power_dbm: float | None) -> float:
    if (snr_db is None) == (tx_power_dbm is None):
        raise ConfigError("run_link needs exactly one of snr_db (fixed SNR) or tx_power_dbm (link budget)")
    if snr_db is not None:
        return noise_variance_for_snr(snr_db)
    return noise_variance_for_tx_power(
        tx_power_dbm=tx_power_dbm, bandwidth_hz=config.bandwidth_hz, noise_figure_db=config.noise_figure_db
    )


def _validate_paths(paths: ChannelPaths) -> None:
    expected = paths.path_gain.shape
    if len(expected) != 1:
        raise ShapeError("path_gain", (-1,), expected)
    for name in ("path_delay_s", "path_azimuth", "path_zenith", "path_valid"):
        actual = getattr(paths, name).shape
        if actual != expected:
            raise ShapeError(name, expected, actual)


def array_response(
    azimuth: FloatArray | float, zenith: FloatArray | float, config: SystemConfig
) -> ComplexArray:
    return sectored_array_response(
        azimuth,
        zenith,
        n_antennas=config.n_antennas,
        spacing_wavelengths=config.antenna_spacing_wavelengths,
        sector_boresight_azimuths=config.sector_boresight_azimuths,
        downtilt=config.sector_downtilt,
        peak_gain_dbi=config.element_peak_gain_dbi,
        beamwidth=config.element_beamwidth,
        front_to_back_db=config.element_front_to_back_db,
    )


def array_beam_pattern(
    beam_weights: ComplexArray, azimuth: FloatArray, zenith: FloatArray | float, config: SystemConfig
) -> FloatArray:
    return beam_pattern(beam_weights, array_response(azimuth, zenith, config))


def system_codebook(config: SystemConfig) -> ComplexArray:
    return sectored_codebook(
        n_antennas=config.n_antennas,
        n_sectors=config.n_sectors,
        oversampling=config.codebook_oversampling,
        spacing_wavelengths=config.antenna_spacing_wavelengths,
        sector=config.codebook_sector,
    )


def _build_channel(
    paths: ChannelPaths, config: SystemConfig, simulation_mode: SimulationMode, normalize: bool
) -> _Channel:
    delay_s = time_aligned_delays(paths.path_delay_s, paths.path_valid)
    gain = np.where(paths.path_valid, paths.path_gain, 0.0).astype(np.complex64)
    steering = array_response(paths.path_azimuth, paths.path_zenith, config)
    isotropic = np.ones((1, gain.shape[-1]), dtype=np.complex64)
    grid_axes = {"n_subcarriers": config.n_subcarriers, "subcarrier_spacing_hz": config.subcarrier_spacing_hz}
    if normalize:
        gain = unit_power_gain(gain, paths_to_cfr(gain, delay_s, isotropic, **grid_axes))
    isotropic_cfr = paths_to_cfr(gain, delay_s, isotropic, **grid_axes)
    cfr = paths_to_cfr(gain, delay_s, steering, **grid_axes)
    if simulation_mode is SimulationMode.FREQUENCY:
        return _Channel(gain, delay_s, cfr, None, cfr, isotropic_cfr)
    n_taps = n_cir_taps(
        delay_s,
        sample_rate_hz=config.sample_rate_hz,
        n_precursor_taps=config.n_cir_precursor_taps,
        n_postcursor_taps=config.n_cir_postcursor_taps,
    )
    cir = paths_to_cir(
        gain,
        delay_s,
        steering,
        sample_rate_hz=config.sample_rate_hz,
        n_precursor_taps=config.n_cir_precursor_taps,
        n_taps=n_taps,
    )
    return _Channel(
        gain, delay_s, cfr, cir, cir_to_cfr(cir, n_subcarriers=config.n_subcarriers), isotropic_cfr
    )


def _select_beam(
    cfr: ComplexArray,
    codebook: ComplexArray,
    beam_power: FloatArray,
    method: BeamMethod,
    candidate_beams: npt.NDArray[np.int64] | None,
    n_probe_beams: int,
) -> _BeamSelection:
    n_subcarriers = cfr.shape[-1]
    match method:
        case BeamMethod.SWEEP | BeamMethod.AI_TOP_K:
            if candidate_beams is None:
                candidate_beams = np.arange(codebook.shape[0], dtype=np.int64)
            beam_idx = strongest_beam_idx(beam_power, candidate_idx=candidate_beams)
            weights = codebook[beam_idx]
            spread = spread_over_subcarriers(weights, n_subcarriers=n_subcarriers)
            return _BeamSelection(weights, spread, beam_idx, int(candidate_beams.size) + n_probe_beams)
        case BeamMethod.MRT_WIDEBAND:
            weights = wideband_mrt_weights(cfr)
            spread = spread_over_subcarriers(weights, n_subcarriers=n_subcarriers)
            return _BeamSelection(weights, spread, None, 0)
        case BeamMethod.MRT_SUBCARRIER:
            weights = subcarrier_mrt_weights(cfr)
            return _BeamSelection(weights, weights, None, 0)


def _receive_grid(
    grid: ComplexArray,
    channel: _Channel,
    selection: _BeamSelection,
    h_eff_true: ComplexArray,
    config: SystemConfig,
    simulation_mode: SimulationMode,
    noise_variance: float,
    rng: np.random.Generator,
) -> ComplexArray:
    if simulation_mode is SimulationMode.FREQUENCY:
        return apply_frequency_channel(grid, h_eff_true, noise_variance=noise_variance, rng=rng)
    antenna_signals = ofdm_modulate(
        precode_grid(grid, selection.subcarrier_weights), cyclic_prefix_length=config.cyclic_prefix_length
    )
    received_signal = apply_time_channel(antenna_signals, channel.cir, noise_variance=noise_variance, rng=rng)
    return ofdm_demodulate(
        received_signal,
        n_ofdm_symbols=config.n_ofdm_symbols,
        n_subcarriers=config.n_subcarriers,
        cyclic_prefix_length=config.cyclic_prefix_length,
    )


def _channel_knowledge(
    received_grid: ComplexArray,
    known_pilots: ComplexArray,
    mask: npt.NDArray[np.bool_],
    h_eff_true: ComplexArray,
    csi_mode: CsiMode,
    config: SystemConfig,
) -> ComplexArray:
    if csi_mode is CsiMode.PERFECT:
        return np.broadcast_to(h_eff_true, (*received_grid.shape[:-2], h_eff_true.shape[-1]))
    pilot_subcarriers = pilot_subcarrier_idx(
        n_subcarriers=config.n_subcarriers,
        pilot_spacing=config.pilot_spacing,
        pilot_offset=config.pilot_offset,
    )
    interpolation = linear_interpolation_matrix(pilot_subcarriers, n_subcarriers=config.n_subcarriers)
    return interpolate_pilots(ls_estimate(received_grid, known_pilots, mask), interpolation)


def _equalize(
    received_grid: ComplexArray, cfr_used: ComplexArray, equalizer: Equalizer, noise_variance: float
) -> ComplexArray:
    if equalizer is Equalizer.ZF:
        return zf_equalize(received_grid, cfr_used)[0]
    return mmse_equalize(received_grid, cfr_used, noise_variance=noise_variance)[0]


def _plotted(symbols: ComplexArray) -> ComplexArray:
    first_slot = symbols[0]
    stride = max(1, -(-first_slot.size // MAX_PLOTTED_SYMBOLS))
    return first_slot[::stride][:MAX_PLOTTED_SYMBOLS]


def run_link(
    paths: ChannelPaths,
    config: SystemConfig,
    *,
    method: BeamMethod,
    modulation: Modulation,
    csi_mode: CsiMode,
    simulation_mode: SimulationMode,
    seed: int,
    n_slots: int,
    snr_db: float | None = None,
    tx_power_dbm: float | None = None,
    candidate_beams: npt.NDArray[np.int64] | None = None,
    n_probe_beams: int = 0,
) -> LinkResult:
    _validate_paths(paths)
    if method is BeamMethod.AI_TOP_K and (candidate_beams is None or candidate_beams.size == 0):
        raise ConfigError("ai_topk needs a non-empty candidate_beams array")
    noise_variance = _noise_variance(config, snr_db, tx_power_dbm)
    rng = np.random.default_rng(seed)
    channel = _build_channel(paths, config, simulation_mode, normalize=snr_db is not None)
    codebook = system_codebook(config)
    beam_power = sweep_beams(channel.cfr, codebook)
    selection = _select_beam(channel.cfr, codebook, beam_power, method, candidate_beams, n_probe_beams)
    h_eff_true = effective_channel(channel.cfr_seen_by_ofdm, selection.subcarrier_weights)
    chosen_power = wideband_power(effective_channel(channel.cfr, selection.subcarrier_weights))
    transmission = _transmission(config, modulation, n_slots, rng)
    received_grid = _receive_grid(
        transmission.grid, channel, selection, h_eff_true, config, simulation_mode, noise_variance, rng
    )
    cfr_used = _channel_knowledge(
        received_grid, transmission.known_pilots, transmission.mask, h_eff_true, csi_mode, config
    )
    equalized_grid = _equalize(received_grid, cfr_used, config.equalizer, noise_variance)
    equalized_symbols = extract_data_symbols(equalized_grid, transmission.mask)
    received_bits = demap_hard(equalized_symbols, modulation=modulation)
    snr = effective_snr(h_eff_true, noise_variance=noise_variance)
    pdp_delay_s, pdp = power_delay_profile(channel.path_gain, channel.path_delay_s)
    return LinkResult(
        method=method,
        beam_idx=selection.beam_idx,
        serving_sector=serving_sector(selection.beam_weights, n_sectors=config.n_sectors),
        beams_tested=selection.beams_tested,
        beam_weights=selection.beam_weights,
        snr_db=float(linear_to_db(snr)),
        beamforming_gain_db=float(linear_to_db(beamforming_gain(chosen_power, channel.isotropic_cfr))),
        ber=bit_error_rate(transmission.bits, received_bits),
        n_bits=int(transmission.bits.size),
        spectral_efficiency=spectral_efficiency(snr),
        power_loss_db=float(power_loss_db(np.max(beam_power), chosen_power)),
        h_eff_true=h_eff_true,
        h_eff_estimated=np.asarray(cfr_used)[0],
        symbols_before_equalization=_plotted(extract_data_symbols(received_grid, transmission.mask)),
        symbols_after_equalization=_plotted(equalized_symbols),
        pdp_delay_s=pdp_delay_s,
        pdp=pdp,
        cfr_reference_antenna=channel.isotropic_cfr[0],
    )
