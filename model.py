"""
Contains the MFModel class for predicting movie ratings by users based on implicit feedback data.

Classes

    MFModel
        Class implementing a Matrix Factorisation model, using explicit feedback movie rating data.

Functions

    global_parallel_metrics(block, U, V, lamda, tau):
        Global metrics calculation function for use in multiprocessing

    global_parallel_update(args, lamda, tau, features, U):
        Global update function for use in multiprocessing

"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import logging
from multiprocessing import Pool
import multiprocessing
from datastructures import dual as ds
import time


class MFModel:
    """
    Class implementing a Matrix Factorisation model, using explicit feedback movie rating data.  Training involves factorising the sparse
    ratings matrix R as the product user feature vectors (U) and item/movie feature vectors (V).  After training, ratings for unseen movies may be
    predicted and used to suggest unseen movies that a user is likely to enjoy.

    Attributes
    ----------
        features
            The dimension of the features vectors describing user preference and item characteristics.
        item_count
            Number of item vectors (also the number of unique items in the data set).
        user_count
            Number of user vectors (also the number of unique users in the data set).
        U
            Matrix of size (user_count x features+1) with rows containing user feature vectors.
        V
            Matrix of size (item_count x features+1) with rows containing item feature vectors.
        I_star
            The identity matrix of size (features+1 x features+1) with the element in the last row and column set to 0.

    Methods
    -------

        get_user_feature_vec(ratings, items_rated, data_set, lamda, tau):
            Computes and returns a single user feature vector given ratings of some items.

        compute_log_loss(self, U, V, lamda, tau, data):
            Computes and returns the log loss for given matrices of user and item vectors, U and V respectively.

        compute_metrics_parallel(self, U, V, lamda, tau, data):
            Computes and returns the log loss and prediction error, using multiprocessing to reduce processing time.

        plot_training(loss, error)
            Plots the training loss and RMSE vs training iteration.

        recommend_from_user_vector(u,already_seen,data_struct,itemBias,minRatings,topk)
            Returns topk item recommendations for a given user feature vector u

        recommend_for_user(userId, data_struct, itemBias, minRatings, topk)
            Returns topk item recommendations for a given user in the data set

        train(data,lamda,tau,max_iter,verbose,plot,continue_training,order,scale_UV_init)
            Trains the model by alternating least squares.

        update(U, lamda, tau, data, count, get_func)
            An abstract function that returns an updated U or V, depending on the get_func and count arguments.

        update_U_parallel(self, V, lamda, tau, data):
            Updates user vectors using multiprocessing for faster processing time.

        update_parallel(U, lamda, tau, data, count, get_func):
            An abstract method which updates user or item feature vectors using multiprocessing for faster processing time.

        update_V_parallel(self, U, lamda, tau, data):
            Updates item vectors using multiprocessing for faster processing time.

        load(path)
            Loads parameters from a saved model at path.

        save(path)
            Saves model parameters.

    """

    def load(self, path="als.pkl"):
        """Loads the parameters from a saved model at path."""
        with open(path, "rb") as file:
            load_dict = pickle.load(file)
        self.V = load_dict["V"]
        self.U = load_dict["U"]
        self.features = load_dict["features"]
        self.item_count = load_dict["item_count"]
        self.user_count = load_dict["user_count"]

    def save(self, path="als.pkl"):
        """Saves model parameters."""
        save_dict = {
            "V": self.V,
            "U": self.U,
            "features": self.features,
            "user_count": self.user_count,
            "item_count": self.item_count,
        }
        with open(path, "wb") as file:
            pickle.dump(save_dict, file)

    def __init__(self, features=10):
        self.features = features
        self.I_star = np.identity(features + 1)
        self.I_star[-1, -1] = 0
        self.user_count = 0
        self.item_count = 0

    def compute_metrics_parallel(self, U, V, lamda, tau, data):
        """Computes and returns the log loss and prediction error, using multiprocessing to reduce processing time."""
        cpu_count = multiprocessing.cpu_count()

        blocks = data.split_structure_items(cpu_count)

        total_MAP = 0
        total_error = 0
        errors = np.zeros(data.get_item_count())

        with Pool(processes=cpu_count) as pool:
            for i, tupOut in enumerate(
                pool.starmap(
                    global_parallel_metrics,
                    [(block, U, V, lamda, tau) for block in blocks],
                )
            ):
                MAP, error = tupOut
                total_MAP += MAP
                total_error += error
                errors[i] = error[i] + error

        # E = error / data.itemIdx["item_count"]
        # plt.plot(data.itemIdx["item_count"], errors, ".k")
        # plt.plot(data.itemIdx["item_count"][7750], E[7750], "or")
        # plt.title("The Napoleon Dynamite Effect")
        # plt.xlabel("rating counts")
        # plt.ylabel("RMSE")
        # plt.show()

        return -total_MAP / data.get_user_count(), np.sqrt(
            total_error / data.get_user_count()
        )

    def compute_log_loss(self, U, V, lamda, tau, data):
        """Computes and returns the log loss for given matrices of user and item vectors, U and V respectively."""
        MAP = 0
        for i, (ui, ri, vi) in enumerate(
            zip(
                data.user2Rating["userId"],
                data.user2Rating["ratings"],
                data.user2Rating["itemId"],
            )
        ):
            MAP += (
                -lamda
                / 2
                * (ri - (np.inner(U[ui, :-1], V[vi, :-1]) + U[ui, -1] + V[vi, -1])) ** 2
                - tau / 2 * np.inner(V[vi, :-1], V[vi, :-1])
                - tau / 2 * np.inner(U[ui, :-1], U[ui, :-1])
            )
        return -MAP / data.get_user_count()

    def update_V(self, U, lamda, tau, data):
        return self.update(U, lamda, tau, data, self.item_count, data.get_item)

    def update_U(self, V, lamda, tau, data):
        return self.update(V, lamda, tau, data, self.user_count, data.get_user)

    def update(self, U, lamda, tau, data, count, get_func):
        """
        An abstract function that returns an updated U or V, depending on the get_func and count arguments.
        """
        V = np.zeros((count, self.features + 1))  # item_count
        for n in range(count):  # item_count
            H = np.zeros((self.features + 1, self.features + 1))
            ratings, who_rated = get_func(n)  # get_item

            Y = np.array(U[who_rated, :])
            b_u = np.array(Y[:, -1, np.newaxis])
            assert b_u.shape[1] == 1
            Y[:, -1] = 1

            S = lamda * Y.T @ (np.array(ratings).reshape((-1, 1)) - b_u)

            for m in range(len(who_rated)):
                y_m = Y[m, :, np.newaxis]
                H = H + lamda * y_m @ y_m.T
                assert H.shape == (self.features + 1, self.features + 1)
            H = H + tau * np.identity(
                self.features + 1
            )  # self.I_star  # np.identity(self.features + 1)  # self.I_star  #

            z = np.linalg.solve(H, S)
            V[n, :] = np.squeeze(z)
        return V

    def get_user_feature_vec(self, ratings, items_rated, data_set, lamda=1, tau=0.01):
        """Computes and returns a single user feature vector given that the argument items_rated received argument ratings from the user."""

        def provided_data(n):
            return np.array(ratings), np.array(items_rated)

        return self.update(self.V, lamda, tau, data_set, 1, provided_data)

    def train(
        self,
        data,
        lamda=1,
        tau=0.01,
        max_iter=50,
        verbose=False,
        plot=False,
        continue_training=False,
        order="vu",
        scale_UV_init=0.01,
    ):
        """
        Trains the model by alternating least squares.

        Parameters:
            data,
                A DualDataStructure object
            lamda=1,
                Regularisation coefficient influencing variance of predicted ratings.
            tau=0.01,
                Regularisation coefficient for user and feature matrices as well as biases.
            max_iter=50,
                Train for max_iter iterations, each of which comprises updating U and V once.
            verbose=False,
                Print detailed information at each epoch.
            plot=False,
                Plot the training loss and errors after training.
            continue_training=False,
                Set true if training is being continued from a previous run.
            order="vu",
                Specifies whether to first update item vectors ane then user vectors ("vu") or user then item vectors ("uv")
            scale_UV_init=0.01,
                A scaling factor for initialisations of U and V.
        """

        self.user_count = data.get_user_count()
        self.item_count = data.get_item_count()

        if continue_training:
            # Initialize with the current user and item feature vector matrices
            U = self.U
            V = self.V
        else:
            # Initialize U and V with from a uniform distribution.
            U = np.random.rand(self.user_count, self.features + 1) * scale_UV_init
            V = np.random.rand(self.item_count, self.features + 1) * scale_UV_init
            U[:, -1] = 0
            V[:, -1] = 0

        rmse = []
        losses = []
        for i in range(max_iter):
            # Alternate between updating V (with U held constant) and U (with V held constant)
            if order == "vu":
                V = self.update_V_parallel(U, lamda, tau, data)
                U = self.update_U_parallel(V, lamda, tau, data)
            else:
                U = self.update_U_parallel(V, lamda, tau, data)
                V = self.update_V_parallel(U, lamda, tau, data)

            loss, er = self.compute_metrics_parallel(U, V, lamda, tau, data)
            losses.append(loss)
            rmse.append(er)
            if verbose:
                print(f"Iteration: {i}, Loss: {loss} , RMSE: {er}")

        # Set class attributes
        self.U = U
        self.V = V

        if plot:
            self.plot_training(losses, rmse)

        return losses, rmse

    def plot_training(self, loss, error):
        """Plots the training loss and RMSE vs training iteration."""

        fig, ax1 = plt.subplots()

        ax1.set_xlabel("iteration")
        ax1.set_ylabel("RMSE", color="red")
        ax1.plot(error, label="error", color="red")
        ax1.tick_params(axis="y", labelcolor="red")

        # Adding Twin Axes

        ax2 = ax1.twinx()

        ax2.set_ylabel("loss", color="blue")
        ax2.plot(loss, label="loss", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")
        plt.title("ALS Training Loss & RMSE")
        plt.show()

    def recommend_for_user(
        self, userId, data_struct, itemBias=True, minRatings=10, topk=5
    ):
        """Returns topk item recommendations for a given user in the data set, only items with at least minRatings are considered."""

        ratings, already_seen_items = data_struct.get_user(userId)

        recommendations = self.recommend_from_user_vector(
            self.U[np.newaxis, userId, :],
            already_seen_items,
            data_struct,
            itemBias,
            minRatings,
            topk,
        )

        return recommendations, data_struct.get_item_title(recommendations)

    def recommend_from_user_vector(
        self,
        u,
        already_seen,
        data_struct,
        itemBias=True,
        minRatings=10,
        topk=5,
    ):
        """Returns topk item recommendations for a given user feature vector u, only items with at least minRatings are considered.
        Setting itemBias=False creates a more user specific recommendation by not considering the intrinsic popularity of items."""

        Est_ratings = np.inner(self.V[:, :-1], u[:, :-1])
        if itemBias:
            Est_ratings = Est_ratings + self.V[:, -1, np.newaxis]
        Est_ratings[np.array(already_seen)] = 0

        # TODO: Make a function
        itemId, ratings_count = data_struct.get_all_items_rating_counts()
        invalid_items = itemId[ratings_count < minRatings]
        Est_ratings[invalid_items] = 0

        return np.argsort(np.squeeze(Est_ratings))[-topk:][::-1]

    def update_V_parallel(self, U, lamda, tau, data):
        """Updates item vectors using multiprocessing for faster processing time"""
        a = self.update_parallel(
            U, lamda, tau, data, self.item_count, data.split_structure_items
        )
        return a

    def update_U_parallel(self, V, lamda, tau, data):
        """Updates user vectors using multiprocessing for faster processing time"""

        a = self.update_parallel(
            V, lamda, tau, data, self.user_count, data.split_structure_users
        )
        return a

    def update_parallel(self, U, lamda, tau, data, count, get_func):
        """An abstract method which updates user or item feature vectors using multiprocessing for faster processing time, based on count and get_func."""

        V = np.zeros((count, self.features + 1))
        cpu_count = multiprocessing.cpu_count()

        blocks = get_func(cpu_count)  # data.split_structure_items(cpu_count)
        item_block_splits = [0]
        s = 0
        for i in range(cpu_count):
            item_block_splits.append(np.max(blocks[i][0]) - np.min(blocks[i][0]) + 1)
            s += blocks[i][0].shape[0]
        indices = np.cumsum(item_block_splits)
        with Pool(processes=cpu_count) as pool:
            for i, matrix in enumerate(
                pool.starmap(
                    global_parallel_update,
                    [(block, lamda, tau, self.features, U) for block in blocks],
                )
            ):
                V[indices[i] : indices[i + 1], :] = matrix

        return V


def global_parallel_metrics(block, U, V, lamda, tau):
    """Global metrics calculation function for use in multiprocessing"""
    MAP = 0
    error = 0
    for i, (vi, ri, ui) in enumerate(zip(block[0], block[1], block[2])):
        MAP += (
            -lamda
            / 2
            * (ri - (np.inner(U[ui, :-1], V[vi, :-1]) + U[ui, -1] + V[vi, -1])) ** 2
            - tau / 2 * np.inner(V[vi, :-1], V[vi, :-1])
            - tau / 2 * np.inner(U[ui, :-1], U[ui, :-1])
        )

        error += (ri - (np.inner(U[ui, :-1], V[vi, :-1]) + U[ui, -1] + V[vi, -1])) ** 2

    return (MAP, error)


def global_parallel_update(args, lamda, tau, features, U):
    """Global update function for use in multiprocessing"""

    I_star = np.identity(features + 1)
    I_star[-1, -1] = 0

    itemId, ratings_array, userId = args
    unique_items, unique_items_count = np.unique(itemId, return_counts=True)
    indices = [0] + list(np.cumsum(unique_items_count))

    out = np.zeros((unique_items.shape[0], features + 1))

    for i, (n, count) in enumerate(zip(unique_items, unique_items_count)):

        ratings = ratings_array[indices[i] : indices[i + 1]]
        who_rated = userId[indices[i] : indices[i + 1]]

        Y = np.array(U[who_rated, :])

        H = np.zeros((features + 1, features + 1))

        b_u = np.array(Y[:, -1, np.newaxis])
        assert b_u.shape[1] == 1
        Y[:, -1] = 1

        S = lamda * Y.T @ (np.array(ratings).reshape((-1, 1)) - b_u)  # check dims

        for m in range(Y.shape[0]):
            y_m = Y[m, :, np.newaxis]
            H = H + lamda * y_m @ y_m.T
            assert H.shape == (features + 1, features + 1)
        H = H + tau * np.identity(features + 1)

        z = np.linalg.solve(H, S)
        out[i, :] = np.squeeze(z)
    return out
