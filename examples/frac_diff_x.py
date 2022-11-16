from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fracdiff2 import _fast_fracdiff


def main():
    x = np.arange(0, 30, 1)
    for i, d in enumerate(np.arange(-1, 1.01, 0.01)):
        fracs = _fast_fracdiff(x, d=d)
        a = pd.DataFrame(data=np.transpose([np.array(fracs), x]), columns=['fd(x, d={:.2f})'.format(d), 'x'])
        a.plot(lw=4, color=['red', 'green'])
        Path('img').mkdir(parents=True, exist_ok=True)
        plt.savefig('img/img_{}.png'.format(str(i).zfill(3)))
        plt.close()


if __name__ == '__main__':
    main()
