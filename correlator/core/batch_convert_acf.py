import matplotlib.pyplot as plt
import pandas as pd
from reading.read import read_FCS
from pyjacket.filetools import FileManager

BIN_TIME = 2e-7  # [s]

fm = FileManager(
    r'D:\Data\FCS',
    r'D:\Analysis\FCS',
    CSV_SEP=','
)

def process_folder(fm: FileManager, folder: str):
    fm.rel_path = folder

    for file_name in fm.iter_dir('.pt3'):

        print(f"\n\nProcessing {file_name}")
        file_path = fm.src_path(file_name)

        # == Read file
        fcs = read_FCS(file_path)

        # Visualize time trace at reduced resolution
        time, counts = fcs.bin_trace(0.06067, hertz=True)
        plt.title('Time Trace')
        plt.plot(time, counts/1000)
        plt.ylabel('Intensity [kHz]')
        plt.xlabel('Time [s]')
        fm.savefig(f"{file_name}.png", folder=f'trace')

        # == Bin the time trace data
        # set a smaller bin_time if you want to plot the time trace (e.g. 0.06)
        time, counts = fcs.bin_trace(BIN_TIME, hertz=True)

        # == Autocorrelate on a log-scale
        # len(acf) ~= m + ilog2(N / m)*(m // 2) + 1
        tau, acf = fcs.compute_ACF(counts, time[1] - time[0])
        tau *= 1000  # [ms]
        
        # == Write output to file
        df = pd.DataFrame({
            'tau': tau,  # [ms]
            'acf': acf,
        })

        fm.write_csv(file_name, df, folder='csv')

        # == Visualize ACF
        plt.plot(tau, acf, '.k')
        plt.xscale('log')
        fm.savefig(f"{file_name}.png", folder=f'acf')



if __name__ == "__main__":
    import os

    for folder in reversed(os.listdir(r'D:\Data\FCS\20241120')):
        process_folder(fm, f'20241120\\{folder}')


    print('\nFinished successfully')