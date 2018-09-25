import numpy as np

import pylab as pl


# https://github.com/SimonOuellette35/FractionalDiff/blob/master/question2.py
# from: http://www.mirzatrokic.ca/FILES/codes/fracdiff.py
# small modification: wrapped 2**np.ceil(...) around int()
def fracdiff(x, d):
    T = len(x)
    np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
    k = np.arange(1, T)
    b = (1,) + tuple(np.cumprod((k - d - 1) / k))
    z = (0,) * (np2 - T)
    z1 = b + z
    z2 = tuple(x) + z
    dx = pl.ifft(pl.fft(z1) * pl.fft(z2))
    return np.real(dx[0:T]), b


if __name__ == '__main__':
    n = 1024
    input_sig = np.random.uniform(size=n)
    d = 0.4
    sig, w = fracdiff(x=input_sig, d=d)

    from marco import getWeights

    w2 = getWeights(d, size=n)
    w2 = np.flip(w2.flatten())
    a = 2
    assert np.mean(np.abs(w - w2)) < 1e-6

    from marco import getWeights_FFD

    w3 = getWeights_FFD(d=d, thres=0, lim=1023)
    w3 = np.flip(w3.flatten())
    assert np.mean(np.abs(w - w3)) < 1e-6

    from marco import fracDiff
    import pandas as pd

    sig2 = fracDiff(pd.DataFrame(data=input_sig.T), d)
    sig2 = list(sig2.values())
    assert np.mean(np.abs(sig[n - len(sig2):] - sig2)) < 1e-2
    print(sig2[-1])
    print(sig[-1])
    print(sig[n - len(sig2)])
    print(sig2[0])

    import matplotlib.pyplot as plt

    assert np.mean(np.abs(sig[n - len(sig2):] - sig2)) < 1e-6
    plt.plot(sig[n - len(sig2):])
    plt.plot(sig2)
    plt.show()

    from marco import fracDiff_FFD

    frac_1_sig = fracDiff_FFD(input_sig, d=1, thres=0)
    a = 2
