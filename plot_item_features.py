"""Plots the movie feature vectors in the latent 2D space.

A number of well known movies are also plotted for examinations."""

import numpy as np
import matplotlib.pyplot as plt
import model
import datastructures.dual as ds

if __name__ == "__main__":

    data = ds.load(path="data/data.pkl")

    recommender = model.MFModel(2)

    # The model was trained as follows:
    # loss, rmse = recommender.train(
    #     data,
    #     max_iter=20,
    #     lamda=1,
    #     tau=0.01,
    #     verbose=True,
    #     plot=True,
    #     continue_training=False,
    #     order="vu",
    #     scale_UV_init=0.01,
    # )
    # recommender.save(path="models/als_R2.pkl")

    # The model als_R2 was specifically trained for this task, with feature length = 2.  This model is useful for visualisation but
    # will have a high predictive error due to the low dimensionality of the latent space.
    recommender.load("models/als_R2.pkl")

    # Plotting the 2D feature vectors for all items/movies with more than 10 ratings.

    mask = data.get_all_items_rating_counts()[1] > 10
    plt.plot(recommender.V[mask, 0], recommender.V[mask, 1], ".b", label="Items")

    def plot_item(vid, marker="."):
        """Plots the latent 2D representation of item/movie vid with the specified marker"""
        plt.plot(
            recommender.V[vid, 0],
            recommender.V[vid, 1],
            marker,
            markersize=10,
            label=data.get_item_title(vid),
        )

    plot_item(257, "s")
    plot_item(1166, "s")
    plot_item(1179, "s")
    plot_item(2537, "s")
    plot_item(5270, "s")
    plot_item(9950, "s")

    plot_item(11601, "*")
    plot_item(14824, "*")
    plot_item(4201, "*")
    plot_item(4119, "X")
    plot_item(1939, "X")
    plot_item(10372, "X")
    plot_item(1191, "X")

    plot_item(0, "P")
    plot_item(2579, "P")
    plot_item(14415, "P")
    plot_item(20029, "<")

    plt.title("Item 2D feature representation")
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.legend()
    plt.show()
