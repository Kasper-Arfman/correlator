import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pyjacket.filetools import FileManager

BASE = r'D:\Analysis\FCS\20241120'

fm = FileManager(
    r'D:\Analysis\FCS\20241120',
    r'D:\Analysis\FCS\20241120',
)

acf = pd.DataFrame()
dacf = pd.DataFrame()

# find the sub folders of a dir
first = True

for folder in os.listdir(BASE):
    if not os.path.isdir(os.path.join(BASE, folder)): 
        continue

    # read the .csv of all acfs
    file_name = f"{BASE}\\{folder}.csv"
    df = pd.read_csv(file_name)

    x = df.pop('tau').to_numpy()
    y = df.to_numpy()

    if first:
        acf['tau'] = x
        dacf['tau'] = x
        first = False

    # # make a plot of all acfs
    # plt.plot(x, y, label=df.columns)
    # plt.xscale('log')
    # # plt.legend()
    # plt.show()
    

    # make a normalized plot of all acfs
    i = np.nonzero(x >= 0.001)[0][1]
    y_norm = (y - 1) / (y[i]-1) + 1
    # plt.plot(x, y_norm)
    # plt.xscale('log')
    # plt.show()


    # compute the average ACF
    y_avg = np.average(y_norm, axis=1)
    # plt.plot(x, y_avg)
    # plt.xscale("log")
    # plt.show()

    acf[folder] = y_avg


    # compute the stdev of the ACF
    y_std = np.std(y_norm, axis=1)
    dacf[folder] = y_std

acf.to_csv(f'{BASE}\\test_acf.csv', index=False)
dacf.to_csv(f'{BASE}\\test_dacf.csv', index=False)

print(f"\nFinished successfully")