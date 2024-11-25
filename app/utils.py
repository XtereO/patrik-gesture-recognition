import numpy as np

def reduce_landmarks(items):
    if(items is None):
        return []
    
    acc = []
    for i in items.landmark:
        acc.append([i.x, i.y, i.z])
    return acc

def add_padding(decoded: bytes):
    dtype = np.float32
    element_size = np.dtype(dtype).itemsize
    print("is it?", len(decoded) % element_size != 0, len(decoded))
    if len(decoded) % element_size != 0:
    # Pad the string to make its length a multiple of element size
        padding_needed = element_size - (len(decoded) % element_size)
        decoded += b'\x00' * padding_needed
    return decoded
