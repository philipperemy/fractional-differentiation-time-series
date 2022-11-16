import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fracdiff import frac_diff_ffd
from utils import plot_multi

if __name__ == '__main__':
    df = pd.read_csv('../doc/sp500.csv', index_col=0, parse_dates=True)
    df = df['1993':]

    fractional_returns = frac_diff_ffd(df['Close'].apply(np.log).values, d=0.4, thres=1e-5)
    df['Fractional differentiation FFD'] = fractional_returns
    df['SP500'] = df['Close']
    df = df[['SP500', 'Fractional differentiation FFD']]

    # burn the first 1500, days where the weights are not defined.
    print(df.head())
    plot_multi(df[1500:])
    plt.show()
