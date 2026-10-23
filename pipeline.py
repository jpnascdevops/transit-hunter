import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from transit_hunter.cleaning import flatten, prepare
from transit_hunter.data import download_lightcurve, generate_synthetic
from transit_hunter.model import fit_transit
from transit_hunter.periodogram import bls, lomb_scargle, lomb_scargle_manual
from transit_hunter.physics import physical_parameters


def save(fig, folder, name):
    path = folder / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  figure: {path}")


def step(number, title):
    print(f"\n[{number}] {title}")


def main():
    ap = argparse.ArgumentParser(description="Transit Hunter: from raw flux to planet radius")
    ap.add_argument("--target", default="WASP-18", help="Star name or TIC/KIC ID")
    ap.add_argument("--synthetic", action="store_true", help="Use a simulated planet instead of downloading data")
    ap.add_argument("--sectors", type=int, nargs="*", help="Specific TESS sectors")
    ap.add_argument("--star-radius", type=float, default=1.0, help="Stellar radius in solar radii")
    ap.add_argument("--star-mass", type=float, default=1.0, help="Stellar mass in solar masses")
    ap.add_argument("--teff", type=float, help="Stellar effective temperature (K)")
    ap.add_argument("--max-period", type=float, default=13.0)
    ap.add_argument("--output", default="output")
    args = ap.parse_args()

    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    truth = None

    step(1, "Loading the light curve")
    if args.synthetic:
        raw, truth = generate_synthetic()
        name = "Synthetic planet"
    else:
        raw = download_lightcurve(args.target, sectors=args.sectors)
        name = args.target
    lc = prepare(raw)
    t, f = lc.time.value, lc.flux.value
    print(f"  {t.size} points, {t.max() - t.min():.1f} days of observation")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, f, ".", ms=1, color="k")
    ax.set(xlabel="Time (days)", ylabel="Normalized flux", title=f"{name}: raw light curve")
    save(fig, folder, "1_raw_lightcurve.png")

    step(2, "Lomb-Scargle: searching for sinusoidal variability (rotation, pulsation)")
    freq, power, ls_period, fap = lomb_scargle(t, f, max_period=args.max_period)
    print(f"  dominant period: {ls_period:.4f} d  (false alarm probability: {fap:.2e})")

    binned = lc.bin(time_bin_size=0.02).remove_nans()
    freq_m = np.linspace(freq.min(), freq.max(), 3000)
    power_m = lomb_scargle_manual(binned.time.value, binned.flux.value, freq_m)
    print(f"  manual implementation (binned data): {1 / freq_m[np.argmax(power_m)]:.4f} d")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(1 / freq, power / power.max(), lw=0.8, label="astropy (all points)")
    ax.plot(1 / freq_m, power_m / power_m.max(), lw=0.8, ls="--", label="manual (binned)")
    ax.set(xscale="log", xlabel="Period (days)", ylabel="Normalized power", title="Lomb-Scargle periodogram")
    ax.legend()
    save(fig, folder, "2_lomb_scargle.png")

    step(3, "Removing stellar variability (detrending) and running BLS")
    flat = flatten(lc)
    _, best, bls_model = bls(flat.time.value, flat.flux.value, flat.flux_err.value, max_period=args.max_period)

    mask = bls_model.transit_mask(t, best["period"], 2.5 * best["duration"], best["t0"])
    flat = flatten(lc, mask=mask)
    tf, ff, ef = flat.time.value, flat.flux.value, flat.flux_err.value
    result, best, _ = bls(tf, ff, ef, max_period=args.max_period)

    print(f"  period: {best['period']:.5f} d | depth: {best['depth'] * 1e6:.0f} ppm | "
          f"duration: {best['duration'] * 24:.2f} h | SNR: {best['snr']:.1f} | transits: {best['n_transits']}")
    even, odd = best["depth_even"], best["depth_odd"]
    verdict = "consistent" if abs(even - odd) < 0.2 * max(even, odd) else "SUSPICIOUS: possible eclipsing binary"
    print(f"  odd/even test: {even * 1e6:.0f} vs {odd * 1e6:.0f} ppm ({verdict})")

    fig, axs = plt.subplots(2, 1, figsize=(12, 7))
    axs[0].plot(tf, ff, ".", ms=1, color="k")
    axs[0].set(xlabel="Time (days)", ylabel="Flux", title="Flattened light curve (stellar variability removed)")
    axs[1].plot(result.period, result.power, lw=0.6, color="k")
    axs[1].axvline(best["period"], color="tab:red", alpha=0.4, lw=4)
    axs[1].set(xscale="log", xlabel="Period (days)", ylabel="SNR", title="BLS periodogram")
    save(fig, folder, "3_bls.png")

    step(4, "Fitting the physical transit model (Mandel & Agol via batman)")
    fit, x, y, curve, res = fit_transit(tf, ff, ef, best["period"], best["t0"], best["duration"], best["depth"])
    print(f"  P     = {fit['period']:.5f} ± {fit['period_err']:.5f} d")
    print(f"  Rp/R* = {fit['rp']:.4f} ± {fit['rp_err']:.4f}")
    print(f"  a/R*  = {fit['a']:.2f} ± {fit['a_err']:.2f}")
    print(f"  inc   = {fit['inc']:.2f} ± {fit['inc_err']:.2f} deg")
    print(f"  reduced chi2 = {fit['reduced_chi2']:.2f}")

    hours = x * 24
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axs[0].plot(hours, y, ".", ms=2, color="0.6", label="data")
    edges = np.linspace(hours.min(), hours.max(), 80)
    idx = np.digitize(hours, edges)
    means = [(hours[idx == i].mean(), y[idx == i].mean()) for i in np.unique(idx) if np.sum(idx == i) > 5]
    axs[0].plot(*zip(*means), "o", ms=4, color="k", label="binned mean")
    axs[0].plot(hours, curve, color="tab:red", lw=2, label="model")
    axs[0].set(ylabel="Flux", title="Phase-folded transit")
    axs[0].legend()
    axs[1].plot(hours, res * 1e6, ".", ms=2, color="0.5")
    axs[1].axhline(0, color="tab:red")
    axs[1].set(xlabel="Hours from mid-transit", ylabel="Residual (ppm)")
    save(fig, folder, "4_fit.png")

    step(5, "Converting to physical quantities")
    phys = physical_parameters(fit, args.star_radius, args.star_mass, args.teff)
    print(f"  planet radius: {phys['planet_radius_earth']:.2f} R_Earth ({phys['planet_radius_jupiter']:.2f} R_Jupiter)")
    print(f"  semi-major axis: {phys['a_kepler_au']:.4f} AU (Kepler) vs {phys['a_fit_au']:.4f} AU (fit)")
    print(f"  implied stellar density: {phys['stellar_density_g_cm3']:.2f} g/cm3 (Sun: 1.41)")
    print(f"  impact parameter b: {phys['impact_parameter']:.2f}")
    if "equilibrium_temp_k" in phys:
        print(f"  equilibrium temperature: {phys['equilibrium_temp_k']:.0f} K")

    report = {"target": name, "lomb_scargle_period": ls_period, "bls": best, "fit": fit, "physics": phys}
    if truth:
        report["truth"] = truth
        print("\n  comparison with injected truth:")
        for key in ("period", "rp", "a", "inc"):
            print(f"    {key:8s} injected {truth[key]:>8.4f} | recovered {fit[key]:>8.4f}")
    (folder / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nResults saved to {folder / 'results.json'}")


if __name__ == "__main__":
    main()
