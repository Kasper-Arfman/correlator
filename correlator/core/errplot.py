
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_err(x, y, dy, color=None, alpha=0.2, **kw):
    line, = plt.plot(x, y, c=color, **kw)
    facecolor = color if color is not None else line.get_color()
    plt.fill_between(x, y-dy, y+dy, alpha=alpha, facecolor=facecolor)

def average_stdev(x: np.ndarray):
    n = x.shape[0]
    var = np.square(x)
    return np.sqrt(np.sum(var, axis=0)) / n



acf = pd.read_csv(r'D:\Analysis\FCS\20241120\test_acf.csv')
dacf = pd.read_csv(r'D:\Analysis\FCS\20241120\test_dacf.csv')


y = pd.DataFrame()
dy = pd.DataFrame()

d = {
    'R': ('01_R110',),
    'FL': ('02_FL100_1', '02_FL100_2', '02_FL100_3'),
    'DM': ('03_DM100_1', '03_DM100_2', '03_DM100_3'),
    'DBD': ('04_DBD100_1', '04_DBD100_2'),
    'DP': ('05_DP100_1', '05_DP100_2', '05_DP100_3'),
    'mNG': ('06_mNG100_1', '06_mNG100_2', '06_mNG100_3'),
}

for k, v in d.items():
    y[k] = np.mean(np.array([acf[c] for c in v]), axis=0)
    dy[k] = average_stdev(np.array([dacf[c] for c in v]))





for c in [
    # 'R',
    'mNG',
    'DBD',
    'DM',
    'DP',
    # 'FL',
        ]:
    
    

    plot_err(acf.tau, y[c], dy[c], label=c)

plt.xscale('log')
plt.xlim([1e-3, 1e3])
plt.ylim([1, 2])
plt.xlabel('Lag Time [ms]')
plt.ylabel('Normalized Autocorrelation [-]')
plt.legend()
plt.show()


