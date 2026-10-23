import numpy as np


def prepare(lc):
    lc = lc.remove_nans().normalize()
    return lc.remove_outliers(sigma_upper=4, sigma_lower=50)


def flatten(lc, window_days=1.0, mask=None):
    cadence = np.nanmedian(np.diff(lc.time.value))
    window = int(window_days / cadence) | 1
    return lc.flatten(window_length=window, mask=mask)
