import numpy as np
import batman
import lightkurve as lk


def download_lightcurve(target, mission="TESS", author="SPOC", exptime=120, sectors=None):
    search = lk.search_lightcurve(target, mission=mission, author=author, exptime=exptime, sector=sectors)
    if len(search) == 0:
        raise ValueError(f"No light curve found for '{target}' ({mission}/{author}, {exptime}s)")
    print(search)
    collection = search.download_all(flux_column="pdcsap_flux")
    return collection.stitch()


def generate_synthetic(
    period=3.2,
    rp=0.10,
    a=9.0,
    inc=88.0,
    t0=1.5,
    days=27.0,
    cadence_min=2.0,
    noise_ppm=800.0,
    rotation_days=5.3,
    rotation_amp=0.005,
    seed=42,
):
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, cadence_min / 1440.0)
    mid = days / 2
    t = t[(t < mid - 0.5) | (t > mid + 0.5)]

    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = t0, period, rp, a, inc
    p.ecc, p.w = 0.0, 90.0
    p.limb_dark, p.u = "quadratic", [0.4, 0.25]
    transit = batman.TransitModel(p, t).light_curve(p)

    rotation = 1 + rotation_amp * np.sin(2 * np.pi * t / rotation_days)
    noise = rng.normal(0, noise_ppm * 1e-6, t.size)
    flux = transit * rotation + noise

    truth = {"period": period, "rp": rp, "a": a, "inc": inc, "t0": t0, "rotation_days": rotation_days}
    lc = lk.LightCurve(time=t, flux=flux, flux_err=np.full(t.size, noise_ppm * 1e-6))
    return lc, truth
