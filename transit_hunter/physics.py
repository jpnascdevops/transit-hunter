import numpy as np
from astropy import constants as c
from astropy import units as u


def physical_parameters(fit, star_radius=1.0, star_mass=1.0, teff=None, albedo=0.3):
    period = fit["period"] * u.day
    r_star = star_radius * u.R_sun
    m_star = star_mass * u.M_sun

    planet_radius = fit["rp"] * r_star
    a_kepler = ((c.G * m_star * period ** 2) / (4 * np.pi ** 2)) ** (1 / 3)
    a_fit = fit["a"] * r_star
    density = 3 * np.pi * fit["a"] ** 3 / (c.G * period ** 2)
    b = fit["a"] * np.cos(np.radians(fit["inc"]))

    k = fit["rp"]
    sine = np.sqrt(max((1 + k) ** 2 - b ** 2, 0)) / (fit["a"] * np.sin(np.radians(fit["inc"])))
    total_duration = (period / np.pi * np.arcsin(min(sine, 1.0))).to(u.hour)

    out = {
        "planet_radius_earth": planet_radius.to(u.R_earth).value,
        "planet_radius_jupiter": planet_radius.to(u.R_jup).value,
        "a_kepler_au": a_kepler.to(u.au).value,
        "a_fit_au": a_fit.to(u.au).value,
        "stellar_density_g_cm3": density.to(u.g / u.cm ** 3).value,
        "impact_parameter": float(b),
        "total_duration_h": total_duration.value,
    }
    if teff:
        out["equilibrium_temp_k"] = teff * np.sqrt(1 / (2 * fit["a"])) * (1 - albedo) ** 0.25
    return out
