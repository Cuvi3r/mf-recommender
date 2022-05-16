Matrix Factorisation based Recommender System

The project implements a movie recommender system using explicit feedback in the form of movie ratings.  

Data set

The MovieLense 25M data set is used and is can be downloaded from 
https://files.grouplens.org/datasets/movielens/ml-25m.zip, with full description at https://grouplens.org/datasets/movielens/.

Initial setup

The data set should be unzipped and the resulting ml-25m folder placed in the data folder.

The pre-trained model can be downloaded from https://www.icloud.com/iclouddrive/0d9bjhmaURvK-GsR6qq2RbB4w#als_full_tau01_regbias and should be placed in the models folder.

Example usage:

Loading and preprocessing data
process_ml_25m.py

Train the model on the full ml-25m data set comprising 25 million ratings:
train_ml_25m.py

Generate recommendations for a user in the data set:
recommend_for_user.py


Test scripts

The integrity of the data preprocessing and loading into the custom data structure can be verified with:
test/test_datastruct_integrity.py

A sanity check for recommendations produced by the trained model can use:
test/test_recommendation.py
Sample output:
test/test_recommendation_output.txt


References

Koenigstein, N., Nice, N., Paquet, U. and Schleyen, N., 2012, September. The Xbox recommender system. In Proceedings of the sixth ACM conference on Recommender systems (pp. 281-284).

Koren, Y., Bell, R. and Volinsky, C., 2009. Matrix factorization techniques for recommender systems. Computer, 42(8), pp.30-37.


