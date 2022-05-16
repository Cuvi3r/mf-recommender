"""A simple sanity check, that if user's feature vector is based on having rated a single movie wiht 5.0,
then his/her top 20 recommended movies should be quite similar to that movie and genre."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
import model
import datastructures.dual as ds

data = ds.load(path="../data/data.pkl")

recommender = model.MFModel(50)
recommender.load("../models/als_full_tau01_regbias.pkl")  # path="als_full_R2.pkl")

# MovidId - Movie Title
# 4201 - Shrek (2001)
# 4887 - Lord of the Rings: The Fellowship of the Ring, The (2001)
# 7923 - Spider-Man (2002)
# 6751 - Kill Bill: Vol. 1 (2003)
# 12821 - Twilight (2008)

# Multiple movieId's can be added to the watched list below with corresponding ratings in the ratings list:

watched = [6751]
ratings = [5.0]

u1 = recommender.get_user_feature_vec(ratings, watched, data, tau=0.01)  # 0.001
rec = recommender.recommend_from_user_vector(
    u1, watched, data, itemBias=False, topk=20, minRatings=10
)

print("For a user who watched\n")
for w in watched:
    print(data.get_item_title(w))
print("\nRecommended:\n")
for s in rec:
    print(data.get_item_title(s))
