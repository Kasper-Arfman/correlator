import numpy as np

class Reader:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.arrival_time = self.read_arrival_time()

    def autocorrelate(self):
        ...

    def binned_trace(self):
        time = ...
        intensity = ...
        return time, intensity
    
    def read_arrival_time(self):
        ...

