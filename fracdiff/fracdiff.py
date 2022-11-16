import numpy as np


# from: http://www.mirzatrokic.ca/FILES/codes/fracdiff.py
# small modification: wrapped 2**np.ceil(...) around int()
# https://github.com/SimonOuellette35/FractionalDiff/blob/master/question2.py
def _fast_fracdiff(x, d):
    import pylab as pl
    T = len(x)
    np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
    k = np.arange(1, T)
    b = (1,) + tuple(np.cumprod((k - d - 1) / k))
    z = (0,) * (np2 - T)
    z1 = b + z
    z2 = tuple(x) + z
    dx = pl.ifft(pl.fft(z1) * pl.fft(z2))
    return np.real(dx[0:T])


def _get_weights(d, size):
    # thres>0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def _fracDiff_original_impl(series, d, thres=.01):
    # 1) Compute weights for the longest series
    w = _get_weights(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    # 3) Apply weights to values
    # df = {}
    output = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue  # exclude NAs
            output[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
        # df[name] = df_.copy(deep=True)
    # df = pd.concat(df, axis=1)
    return output


def _get_weight_ffd(d, thres, lim):
    w, k = [1.], 1
    ctr = 0
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        ctr += 1
        if ctr == lim - 1:
            break
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def _fracDiff_FFD_original_impl(series, d, thres=1e-5):
    import pandas as pd
    # 1) Compute weights for the longest series
    w = _get_weight_ffd(d, thres, len(series))
    width = len(w) - 1
    # df = {}
    output = []
    for name in series.columns:
        seriesF, _ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        output.extend([0] * width)
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            # df_[loc1] =
            output.append(np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0])
        # df[name] = df_.copy(deep=True)
    # df = pd.concat(df, axis=1)
    return output


def frac_diff_ffd(x, d, thres=1e-5, disable_warning=False):
    if np.max(x) > 10.0 and not disable_warning:
        print('WARNING: have you applied log before calling this function? If yes, discard this warning.')
    w = _get_weight_ffd(d, thres, len(x))
    width = len(w) - 1
    output = []
    output.extend([0] * width)
    for i in range(width, len(x)):
        output.append(np.dot(w.T, x[i - width:i + 1])[0])
    return np.array(output)
