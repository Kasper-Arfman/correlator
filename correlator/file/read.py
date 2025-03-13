import os

from ..models.reader import Reader
from ._pt3 import FCSreaderPT3, FCSreaderPTU

READERS = {
        'pt3': FCSreaderPT3,
        'ptu': FCSreaderPTU,
}

def read(file_path: str, **kwargs) -> Reader:
    ext = os.path.splitext(file_path)[1].lstrip('.')
    if ext not in READERS:
        raise NotImplementedError(f'Cannot read file of type {ext}')

    reader = READERS[ext]
    return reader(file_path, **kwargs)