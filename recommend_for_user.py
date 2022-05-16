"""Generates and prints a recommendations for users in the MovieLens mk-25m dataset.

Example output can be seen in example_recommendations.txt"""

import numpy as np
import matplotlib.pyplot as plt
from datastructures import dual as ds
import model

# Loading data
data = ds.load(path="data/data.pkl")

# Creating and loading the trained model
recommender = model.MFModel(50)
recommender.load("models/als_full_tau01_regbias.pkl")  # path="als_full_R2.pkl")

userid, ucount = data.get_all_users_rating_counts()

# To run for a particular user, set the user here, or uncomment below to choose a random user who rated 20 movies:
user = 132202
# user = np.random.choice(userid[ucount == 20])

# Get recommendation for user
rec, rec_titles = recommender.recommend_for_user(
    user, data, itemBias=True, minRatings=5, topk=20
)

# Fetch already viewed movies and titles
ratings, watched_itemid = data.get_user(user)
watched_titles = data.get_item_title(watched_itemid)


print(f"For userid: {user}, who watched\n")
for w in watched_titles:
    print(w)
    # print(data.get_item_title(w))
print("\nRecommended:\n")
for s in rec_titles:
    print(s)
