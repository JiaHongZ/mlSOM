import numpy as np
import pickle

with open('knowledge.pkl', 'rb') as f:
    results = pickle.load(f)

print(results.shape)
