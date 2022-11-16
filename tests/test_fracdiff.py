import unittest

import numpy as np


class FracDiffTest(unittest.TestCase):

    def test_limits(self):
        n = 1024
        input_sig = np.log(np.cumsum(np.random.uniform(size=n)))
        diff = np.diff(input_sig)
        from fracdiff2 import frac_diff_ffd
        returns_1 = frac_diff_ffd(x=input_sig, d=1)
        returns_0 = frac_diff_ffd(x=input_sig, d=0)
        returns_05 = frac_diff_ffd(x=input_sig, d=0.5)

        assert np.mean(np.abs(returns_0 - input_sig)) < 1e-6
        assert np.mean(np.abs(returns_1[1:] - diff)) < 1e-6
        assert np.mean(np.abs(returns_05 - input_sig)) > 1e-2

    def test_frac_diff(self):
        n = 1024
        input_sig = np.random.uniform(size=n)
        d = 0.4
        from fracdiff2 import _fast_fracdiff

        sig = _fast_fracdiff(x=input_sig, d=d)

        from fracdiff2 import _get_weights

        w2 = _get_weights(d, size=n)
        # w2 = np.flip(w2.flatten())
        # assert np.mean(np.abs(w - w2)) < 1e-6

        from fracdiff2 import _get_weight_ffd

        w3 = _get_weight_ffd(d=d, thres=0, lim=1024)
        # w3 = np.flip(w3.flatten())
        # assert np.mean(np.abs(w - w3)) < 1e-6

        from fracdiff2 import _fracDiff_original_impl
        import pandas as pd

        sig2 = _fracDiff_original_impl(pd.DataFrame(data=input_sig.T), d)
        sig2 = list(sig2.values())
        assert np.mean(np.abs(sig[n - len(sig2):] - sig2)) < 1e-2
        print(sig2[-1])
        print(sig[-1])
        print(sig[n - len(sig2)])
        print(sig2[0])

        # import matplotlib.pyplot as plt

        assert np.mean(np.abs(sig[n - len(sig2):] - sig2)) < 1e-6
        # plt.plot(sig[n - len(sig2):])
        # plt.plot(sig2)
        # plt.show()

        from fracdiff2 import frac_diff_ffd, _fracDiff_FFD_original_impl

        frac_0_sig = _fracDiff_FFD_original_impl(pd.DataFrame(input_sig), d=1, thres=0.01)
        frac_1_sig = frac_diff_ffd(input_sig, d=1, thres=0.01)
        assert np.mean(np.abs(np.array(frac_0_sig) - np.array(frac_1_sig))) < 1e-6

        frac_0_sig = _fracDiff_FFD_original_impl(pd.DataFrame(input_sig), d=0.4, thres=1e-4)
        frac_1_sig = frac_diff_ffd(input_sig, d=0.4, thres=1e-4)
        assert np.mean(np.abs(np.array(frac_0_sig) - np.array(frac_1_sig))) < 1e-6
        assert len(frac_0_sig) == len(input_sig)
