import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from fracdiff import fast_fracdiff

if __name__ == '__main__':
    x = np.arange(0, 30, 1)

    for i, d in enumerate(np.arange(-1, 1.01, 0.01)):
        fracs = fast_fracdiff(x, d=d)
        a = pd.DataFrame(data=np.transpose([np.array(fracs), x]),
                         columns=['fd(x, d={:.2f})'.format(d), 'x'])
        a.plot(lw=4, color=['red', 'green'])
        if not os.path.exists('img'):
            os.makedirs('img')
        plt.savefig('img/img_{}.png'.format(str(i).zfill(3)))
        plt.close()
        # plt.show()
