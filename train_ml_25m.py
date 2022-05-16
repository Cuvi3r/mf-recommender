"""Trains a model on the full Movielens ml-25m dataset and save the resulting model."""

import numpy as np
import matplotlib.pyplot as plt
import model
import datastructures.dual as ds

if __name__ == "__main__":

    ratings_path = "data/ml-25m/ratings.csv"
    items_path = "data/ml-25m/movies.csv"

    # data.readRawData(ratings_path,items_path)
    # ds.save(data)
    data = ds.load(path="data/data.pkl")

    # The model used 50 latent dimensions for user and item feature vectors
    recommender = model.MFModel(50)

    loss, rmse = recommender.train(
        data,
        max_iter=30,
        lamda=1,
        tau=0.01,
        verbose=True,
        plot=True,
        continue_training=False,
        order="vu",
    )

    # Save the model
    recommender.save(path="models/als_tau01_regbias.pkl")

    # Saving training metrics
    np.savetxt("models/loss_tau01_regbias.txt", loss)
    np.savetxt("models/rmse_tau01_regbias.txt", rmse)
