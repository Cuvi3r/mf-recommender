"""Contains the DualDataStructure and save and load helper functions.

Classes:

    DualDataStructure
        Data structure comprising two copies of the ratings data, one sorted by user index and the 
        other by item index.  This allows rapid sequencial reading of all ratings for a referenced user
        or item.

Functions:

    load(path) -> DualDataStructure
        Unpickles and loads a data structure from the file at path.

    save(ds, path)
        Pickles and stores an entire instance of the data structure to file.

Misc variables:

    SORT_BY_USER tuple
    SORT_BY_ITEM tuple

"""

import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import datastructures.base as ds

SORT_BY_USER = (0, 1, 2)
SORT_BY_ITEM = (2, 1, 0)


def load(path="data/data.pkl"):
    """Unpickles and loads a data structure from the file at path."""
    with open(path, "rb") as file:
        pkl_dict = pickle.load(file)
    return pkl_dict["ds"]


def save(ds, path="data/data.pkl"):
    """Pickles and stores an entire instance of the data structure to file."""
    pkl_dict = {
        "ds": ds,
    }
    with open(path, "wb") as file:
        pickle.dump(pkl_dict, file)


class DualDataStructure:
    """Data structure comprising two copies of the ratings data, one sorted by user index and the
    other by item index.  This allows rapid sequencial reading of all ratings for a referenced user
    or item.

    Attribubes
    ----------

        user_sorted_ds
            Instance of datastructures.BaseStruct, indexed by userid.
        item_sorted_ds
            Instance of datastructures.BaseStruct, indexed by itemid.

    Methods
    -------

        get_all_items_rating_counts() -> np.array
        get_all_users_rating_counts() -> np.array
        get_rating_count() -> int
        get_user_count() -> int
        get_item_count() -> int
        get_item_title(indices) -> list
        get_user(idx) -> tuple
        get_item(idx) -> tuple
        split_structure_items(n) -> list
        split_structure_users(n) -> list

    """

    def __init__(self, ratings_path, item_titles_path):
        """Creates two BaseStruct objects.  Identical ratings data are loaded into both, but one is sorted by
        user index and the other by item index.  This has an advantage for rapid sequential reading of
        ratings as dereferenced by item or user.  Titles are also loaded from file."""
        self.user_sorted_ds = ds.BaseStruct(ratings_path, col_order=SORT_BY_USER)
        self.item_sorted_ds = ds.BaseStruct(ratings_path, col_order=SORT_BY_ITEM)
        self.item_sorted_ds.load_descriptions(item_titles_path)

        assert (
            self.user_sorted_ds.get_rating_count()
            == self.item_sorted_ds.get_rating_count()
        )

    def get_all_items_rating_counts(self):
        """Returns a tuple comprising an array of unique items and the number of ratings for each of them."""
        return self.item_sorted_ds.get_rating_counts_for_primary_keys()

    def get_all_users_rating_counts(self):
        """Returns a tuple comprising an array of unique users and the number of ratings by each of them."""
        return self.user_sorted_ds.get_rating_counts_for_primary_keys()

    def get_rating_count(self):
        """Returns the total number of ratings in the data set."""
        return self.user_sorted_ds.get_rating_count()

    def get_user_count(self):
        """Returns the number of unique users in the data set."""
        return self.user_sorted_ds.get_primary_key_count()

    def get_item_count(self):
        """Returns the number of unique items that were rated in the data set"""
        return self.item_sorted_ds.get_primary_key_count()

    def get_item_title(self, indices):
        """Returns an array of item titles corresponding with the supplied array of item indices"""
        return self.item_sorted_ds.get_descriptions(indices)

    def get_user(self, idx):
        """Returns a tuple containing all ratings by user idx and as well as the itemid's of the items rated."""
        return self.user_sorted_ds.get_rating(idx)

    def get_item(self, idx):
        """Returns a tuple containing all ratings of item idx and as well as the userid's who submitted the
        ratings."""
        return self.item_sorted_ds.get_rating(idx)

    def split_structure_items(self, n):
        """Sub-partitions the data set into n subsets with no ratings of an item in more one subset."""
        return self.item_sorted_ds.split_for_multiprocess(n)

    def split_structure_users(self, n):
        """Sub-partitions the data set into n subsets with no ratings by a user in more one subset."""
        return self.user_sorted_ds.split_for_multiprocess(n)
