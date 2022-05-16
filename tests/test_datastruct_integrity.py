"""Verifies the integrity of the data in the DualDataStructure object, thereby testing its preprocessing algorithm."""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datastructures.dual as ds

print("Loading and preprocessing data... ")
ratings_path = "../data/ml-25m/ratings.csv"
items_path = "../data/ml-25m/movies.csv"

data = ds.DualDataStructure(ratings_path, items_path)
# ds.save(data, "../data/test.pkl")
# data = ds.load("../data/test.pkl")

print("Loading and preprocessing data complete. ")
print("Starting tests...")

for i in tqdm(range(data.get_user_count())):
    user_ratings, items = data.get_user(i)
    for j, item in enumerate(items):
        item_ratings, users = data.get_item(item)
        assert i in users
        assert user_ratings[j] == item_ratings[np.where(users == i)[0][0]]

print("Forward cross-check completed successfully!")

for i in tqdm(range(data.get_item_count())):
    item_ratings, users = data.get_item(i)
    for j, user in enumerate(users):
        user_ratings, items = data.get_user(user)
        assert i in items
        assert item_ratings[j] == user_ratings[np.where(items == i)[0][0]]

print("Reverse cross-check completed successfully!")
