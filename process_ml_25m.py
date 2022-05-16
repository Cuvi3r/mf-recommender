"""A simple program to process all the raw ml-25m data and load it into the appropriate data structure."""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import datastructures.dual as ds

ratings_path = "data/ml-25m/ratings.csv"
items_path = "data/ml-25m/movies.csv"

data = ds.DualDataStructure(ratings_path, items_path)

# Uncomment save the resulting data structure
# ds.save(data, "data/data.pkl")
