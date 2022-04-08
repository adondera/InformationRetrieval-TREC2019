import io
import spacy
import pandas as pd
import tqdm
import numpy as np

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    # for line in tqdm.tqdm(fin):
    #     tokens = line.rstrip().split(' ')
    #     data[tokens[0]] = map(float, tokens[1:])
    return data

