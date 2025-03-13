from ..models.reader import Reader

class FCSreaderPT3(Reader):

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.time = self.get_time()
    
    def get_time(self):
        time, _, _, meta = pqreader.load_pt3(self.file_path)
        return time * meta['timestamps_unit']
    

class FCSreaderPTU(Reader):
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.time = self.get_time()
    
    def get_time(self):
        ptu_data = ptufile.PtuFile(self.file_path)
        records = ptu_data.decode_records(ptu_data.read_records())
        return records['time'] * ptu_data.global_resolution