import numpy as np
import batman
from scipy.optimize import curve_fit


def centered_phase(t, period, t0):
    return (t - t0 + 0.5 * period) % period - 0.5 * period


def fit_transit(t, y, dy, period, t0, duration, depth, u=(0.4, 0.25)):
    near = np.abs(centered_phase(t, period, t0)) < 2.5 * duration
    tt, yy, ee = t[near], y[near], dy[near]

    p = batman.TransitParams()
    p.t0, p.per = t0, period
    p.rp = np.sqrt(max(depth, 1e-5))
    p.a = max(period / (np.pi * duration), 2.0)
    p.inc, p.ecc, p.w = 89.0, 0.0, 90.0
    p.limb_dark, p.u = "quadratic", list(u)
    m = batman.TransitModel(p, tt)

    def curve(_, t0_, per, rp, a, inc):
        p.t0, p.per, p.rp, p.a, p.inc = t0_, per, rp, a, inc
        return m.light_curve(p)

    guess = [t0, period, p.rp, p.a, 88.0]
    bounds = (
        [t0 - duration, period * 0.995, 0.001, 1.5, 70.0],
        [t0 + duration, period * 1.005, 0.5, 200.0, 90.0],
    )
    best, cov = curve_fit(curve, tt, yy, p0=guess, sigma=ee, absolute_sigma=True, bounds=bounds, maxfev=20000)
    errors = np.sqrt(np.diag(cov))
    model = curve(tt, *best)
    residuals = yy - model

    names = ["t0", "period", "rp", "a", "inc"]
    fit = {n: float(v) for n, v in zip(names, best)}
    fit.update({f"{n}_err": float(e) for n, e in zip(names, errors)})
    fit["reduced_chi2"] = float(np.sum((residuals / ee) ** 2) / (tt.size - best.size))

    phase = centered_phase(tt, fit["period"], fit["t0"])
    order = np.argsort(phase)
    return fit, phase[order], yy[order], model[order], residuals[order]
