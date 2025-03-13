from phconvert import pqreader
import numpy as np
import os
import ptufile
from correlator.core.acf import bin_trace, compute_ACF


class FCSreader:
    file_path: str
    time: np.ndarray

    def bin_trace(self, bin_time, hertz=True):
        return bin_trace(self.time, bin_time, hertz)

    def compute_ACF(self, binned_trace: np.ndarray, bin_time, m=16):
        return compute_ACF(binned_trace, bin_time, m)


class FCSreaderPT3(FCSreader):

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.time = self.get_time()
    
    def get_time(self):
        time, _, _, meta = pqreader.load_pt3(self.file_path)
        return time * meta['timestamps_unit']
    

class FCSreaderPTU(FCSreader):
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.time = self.get_time()
    
    def get_time(self):
        ptu_data = ptufile.PtuFile(self.file_path)
        records = ptu_data.decode_records(ptu_data.read_records())
        return records['time'] * ptu_data.global_resolution
    


def read_FCS(file_path: str) -> FCSreaderPT3:
    _, ext = os.path.splitext(file_path)
    reader = {
        '.pt3': FCSreaderPT3,
        '.ptu': FCSreaderPTU,
    }.get(ext, None)

    if reader is None:
        raise NotImplementedError(f'Cannot read file of type {ext}')
    return reader(file_path)