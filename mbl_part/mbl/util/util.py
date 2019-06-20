import numpy as np
import os
import pickle

def to_onehot(ac, num_ac):
    if not isinstance(ac, np.ndarray): ac = np.array([ac,]) 
    tmp = np.zeros((len(ac), num_ac))       
    tmp[np.arange(len(ac)).astype(np.int), ac.astype(np.int)] = 1
    return tmp

def normalize(x, rms, epsilon=0):
    return (x - rms.mean) / np.sqrt(rms.var + epsilon)

def denormalize(x, rms, epsilon=0):
    return x * (np.sqrt(rms.var + epsilon)) + rms.mean

def load_extracted_val_data(path):
    with open(path, 'rb') as f:
        val_dataset = pickle.load(f)
    return val_dataset

def load_dumped_val_data(path, size=int(1e4)):
    with open(path, 'rb') as f:
        val_dataset = pickle.load(f)
    n = len(val_dataset)
    idxes = np.arange(n)
    np.random.shuffle(idxes)                    
    idxes = idxes[:size]
    ob = np.array([d[0].copy() for d in val_dataset])[idxes]
    ac = np.array([d[1].copy() for d in val_dataset])[idxes]
    ob_next = np.array([d[2].copy() for d in val_dataset])[idxes]

    del val_dataset    

    return {'ob': ob, 'ac': ac, 'ob_next': ob_next}
