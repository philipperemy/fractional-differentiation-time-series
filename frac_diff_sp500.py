import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from fracdiff import frac_diff_ffd
from utils import plot_multi

if __name__ == '__main__':
    close = pd.read_csv(os.path.join('doc', 'sp500.csv'), index_col=0, parse_dates=True)[['Close']]
    close = close['1993':]

    fracs = frac_diff_ffd(close.apply(np.log), d=0.4, thres=1e-5)
    a = pd.DataFrame(data=np.transpose([np.array(fracs), close['Close'].values]),
                     columns=['Fractional differentiation FFD', 'SP500'])

    # burn the first 1500 days where the weights are not defined.
    plot_multi(a[1500:])
    plt.show()
