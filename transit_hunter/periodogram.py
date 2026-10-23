import numpy as np
from astropy.timeseries import BoxLeastSquares, LombScargle


def lomb_scargle_manual(t, y, frequencies, chunk=400):
    y = y - y.mean()
    var = y.var()
    power = np.empty_like(frequencies)
    for i in range(0, frequencies.size, chunk):
        w = 2 * np.pi * frequencies[i:i + chunk, None]
        tau = np.arctan2(np.sin(2 * w * t).sum(1), np.cos(2 * w * t).sum(1)) / (2 * w[:, 0])
        phase = w * (t - tau[:, None])
        c, s = np.cos(phase), np.sin(phase)
        cos_term = (y * c).sum(1) ** 2 / (c ** 2).sum(1)
        sin_term = (y * s).sum(1) ** 2 / (s ** 2).sum(1)
        power[i:i + chunk] = (cos_term + sin_term) / (2 * var)
    return power


def lomb_scargle(t, y, min_period=0.1, max_period=15.0, samples=20000):
    frequencies = np.linspace(1 / max_period, 1 / min_period, samples)
    ls = LombScargle(t, y)
    power = ls.power(frequencies)
    best = frequencies[np.argmax(power)]
    return frequencies, power, 1 / best, ls.false_alarm_probability(power.max())


def bls(t, y, dy, min_period=0.5, max_period=13.0, durations=None):
    if durations is None:
        durations = np.array([0.03, 0.05, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25])
    model = BoxLeastSquares(t, y, dy)
    periods = np.exp(np.linspace(np.log(min_period), np.log(max_period), 30000))
    result = model.power(periods, durations, objective="snr")
    i = np.argmax(result.power)
    best = {
        "period": float(result.period[i]),
        "t0": float(result.transit_time[i]),
        "duration": float(result.duration[i]),
        "depth": float(result.depth[i]),
        "snr": float(result.depth_snr[i]),
    }
    stats = model.compute_stats(best["period"], best["duration"], best["t0"])
    best["depth_even"] = float(stats["depth_even"][0])
    best["depth_odd"] = float(stats["depth_odd"][0])
    best["n_transits"] = int(np.sum(stats["per_transit_count"] > 0))
    return result, best, model
