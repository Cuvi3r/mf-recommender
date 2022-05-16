"""
Contains a base data structure class.

Classes:
    
    BaseStruct
        Data structure that stores the related ratings of different items by different users,
        as indexed by one of items or users.

"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class BaseStruct:
    """
    Data structure that stores the related ratings of different items by different users, as indexed by
    one of items or users.

    Attributes
    ----------

        ratings_dict
        descriptions_ar
        end_indices_ar
        unique_primary_indices

    Methods
    -------

        get_rating_counts_for_primary_keys() -> np.array
            Returns an array containing rating counts for each if the primary keys.
        get_rating_count() -> int
            Returns the total number of ratings in the dataset
        get_primary_key_count() -> int
            Returns the number of primary keys.
        get_descriptions(primary_keys_ar) -> np.array
            Returns an array of corresponding descriptions for the supplied array of primary keys.
        get_rating(primary_key) -> tuple
            Returns a tuple containing all ratings related to the primary key and also the corresponding
            secondary keys associated with these ratings.
        load_descriptions(path) -> np.array
            Reads descriptions or titles for each primary index from the file at path.
        split_for_multiprocess(n) -> list

    """

    def __init__(self, ratings_path, col_order=[0, 1, 2]):
        """
        Loads and preprocesses the ratings data from the file at ratings_path, reindexing both userid's and
        itemid's to be contiguous and begin at 0.

        col_order = SORT_BY_USER or SORT_BY_ITEM, specifies whether the data is sorted by user or item index
        in this object.
        """

        with open(ratings_path, "r") as file:
            lines = file.readlines()

        line_count = len(lines) - 1

        raw_ratings_data = np.ones((line_count, 3)) * -1

        for i, line in enumerate(tqdm(lines)):
            if i == 0:
                continue
            split_str = line.split(",")
            # Column 0 of raw_ratings_data receives raw user indices.
            raw_ratings_data[i - 1, 0] = float(split_str[0])
            # Column 1 of raw_ratingsdata received the ratings.
            raw_ratings_data[i - 1, 1] = float(split_str[2])
            # Column 2 of raw_ratingsdata receives the raw item indices.
            raw_ratings_data[i - 1, 2] = float(split_str[1])
            """Therefore each row of raw_ratings_data contains one rating: User with index (col0) gave rating 
            (col1) to the item with index (col2)."""

        """The argument col_order[0] specifies which column in raw_ratings is the primary attribute by which the 
        data will be sorted in this instance.  col_order[1] always indicates the rating attribute, and 
        col_order[2] indicates the secondary attribute (not sorted by)"""

        # Finding unique primary indices and their number of occurences in the data.
        self.unique_primary_indices, ucount = np.unique(
            raw_ratings_data[:, col_order[0]], return_counts=True
        )

        # __primary_key_count holds the number of unique primary indices.
        self.__primary_key_count = len(self.unique_primary_indices)

        """__end_indices holds the (last index +1) of the ratings of / by each unique primary key.  This is used
        for sequential reading from the data structure"""
        self.__end_indices_ar = np.cumsum(ucount)

        """Original primary attribute indices are reindexed to start at 0 and to be contiguous.  Corresponding 
        values in raw_ratings_data are overwritten"""
        prim_idx = np.arange(self.__primary_key_count)
        primary_key = np.argsort(raw_ratings_data[:, col_order[0]])
        raw_ratings_data[primary_key, col_order[0]] = np.repeat(prim_idx, ucount)

        """Original secondary attribute indices are reindexed to start at 0 and to be contiguous.  Corresponding 
        values in raw_ratings_data are overwritten"""
        unique_secondary_indices, sec_count = np.unique(
            raw_ratings_data[:, col_order[2]], return_counts=True
        )
        __secondary_key_count = len(unique_secondary_indices)
        sec_idx = np.arange(__secondary_key_count)
        sort_by_secondary_idx = np.argsort(raw_ratings_data[:, col_order[2]])
        raw_ratings_data[sort_by_secondary_idx, col_order[2]] = np.repeat(
            sec_idx, sec_count
        )

        # Data is collected in a dictionary after being sorted by the primary attribute indices.
        self.ratings_dict = {
            "primary_key": raw_ratings_data[primary_key, col_order[0]].astype(int),
            "ratings": raw_ratings_data[primary_key, col_order[1]],
            "secondary_key": raw_ratings_data[primary_key, col_order[2]].astype(int),
        }
        self.__descriptions_ar = None

    def get_rating_counts_for_primary_keys(self):
        """Returns a tuple containing arrays of unique primary keys and the number of ratings associated with
        each"""
        return np.unique(self.ratings_dict["primary_key"], return_counts=True)

    def get_rating_count(self):
        """Returns the total number of ratings in the dataset"""
        return len(self.ratings_dict["ratings"])

    def get_primary_key_count(self):
        """Returns the number of primary keys."""
        return self.__primary_key_count

    def __get_start_end_idx__(self, primary_key):
        # index = self.userIdx["userId"].index(idx)
        if primary_key > 0:
            start = self.__end_indices_ar[primary_key - 1]
        else:
            start = 0
        end = self.__end_indices_ar[primary_key]
        return start, end

    def get_descriptions(self, primary_keys_ar):
        """Returns an array of corresponding descriptions for the supplied array of primary keys."""
        if len(self.descriptions) > 0:
            return self.descriptions[primary_keys_ar]
        else:
            return []

    def get_rating(self, primary_key):
        """Returns a tuple containing all ratings related to the primary key and also the corresponding
        secondary keys associated with these ratings."""
        start, end = self.__get_start_end_idx__(primary_key)
        return (
            self.ratings_dict["ratings"][start:end],
            self.ratings_dict["secondary_key"][start:end],
        )

    def load_descriptions(self, path):
        """Reads descriptions or titles for each primary index from the file at path"""
        file = open(path, "r")
        lines = file.readlines()
        line_count = len(lines) - 1

        #        data = np.ones((line_count, 3)) * -1
        keys_list = []
        titles_list = []
        genres_list = []

        for i, line in enumerate(lines):
            if i == 0:
                continue
            split1 = line.find(",")
            split2 = len(line) - line[::-1].find(",")

            itemId = line[:split1]
            itemTitle = line[split1 + 1 : split2 - 1]
            # Removing double quotes if they are in place
            if itemTitle.find('"') > -1:
                itemTitle = itemTitle[1:-1]
            itemGenre = line[split2:-1]
            keys_list.append(int(itemId))
            titles_list.append(itemTitle)
            genres_list.append(itemGenre)

        # Find and remove titles that are not use in the ratings
        title_keys = np.array(keys_list, dtype=int)
        item_xor = np.setxor1d(
            title_keys, self.unique_primary_indices, assume_unique=True
        )
        for id in item_xor:
            idx = keys_list.index(id)
            del keys_list[idx]
            del titles_list[idx]
            del genres_list[idx]
        item_xor = np.setxor1d(
            keys_list,
            self.unique_primary_indices,
            assume_unique=True,
        )
        assert item_xor.size == 0

        sorted_title_id = np.argsort(keys_list)
        self.descriptions = np.array(titles_list)[sorted_title_id]
        assert self.descriptions.size == self.__primary_key_count

        return self.descriptions

    def split_for_multiprocess(self, n):
        """Splits the data set into n subsets for parallel processing during model training."""
        blocks = []
        total_rating_count = len(self.ratings_dict["ratings"])

        split = total_rating_count // n
        prev_idx = 0
        check = 0
        for i in range(1, n + 1):
            # Finding first occurence of rating of itemId.
            if i == n:
                idx = self.ratings_dict["primary_key"].shape[0]
            else:
                idx = np.where(
                    self.ratings_dict["primary_key"]
                    == self.ratings_dict["primary_key"][i * split]
                )[0][0]
            blocks.append(
                (
                    self.ratings_dict["primary_key"][prev_idx:idx],
                    self.ratings_dict["ratings"][prev_idx:idx],
                    self.ratings_dict["secondary_key"][prev_idx:idx],
                )
            )
            prev_idx = idx
            check += len(blocks[i - 1][0])

        assert check == total_rating_count

        return blocks

    def __write_descriptions__(self, path):
        """Writes the primary key descriptions to file, as well as the rating count associated with teach"""
        if len(self.descriptions) > 0:
            with open(path, "w") as file:
                idx_prev = 0
                for i, string in enumerate(self.descriptions):
                    file.writelines(
                        f"{i}, " + string + f", {self.__end_indices_ar[i]-idx_prev}\n"
                    )
                    idx_prev = self.__end_indices_ar[i]
