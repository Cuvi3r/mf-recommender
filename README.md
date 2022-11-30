# Matrix factorisation based Recommender System

#### Matthew Johl - 25699679
#### Stellenbosch University

Created as part of the Applied Machine Learning at Scale course.

The project implements a movie recommender system using explicit feedback in the form of movie ratings.  

## Data set

The MovieLense 25M data set is used and can be downloaded from https://files.grouplens.org/datasets/movielens/ml-25m.zip, with full description at https://grouplens.org/datasets/movielens/.

## Initial setup

The data set should be downloaded unzipped with the resulting ml-25m folder placed in the ./data folder.

The pre-trained model can be downloaded from https://www.icloud.com/iclouddrive/0d9bjhmaURvK-GsR6qq2RbB4w#als_full_tau01_regbias and should be placed in the ./models folder.

## Example usage:

Loading and preprocessing data:\
python process_ml_25m.py\
During this process the movies and users are reindexed for continquity and to start at 0.  The new movie indices and titles can be found at:\
./data/title_25m.txt

Train the model on the full ml-25m data set comprising 25 million ratings, using the Alternating Least Squares algorithm:\
python train_ml_25m.py\
<img src=https://user-images.githubusercontent.com/103119572/168556922-f5891be0-96f3-4320-943f-1a4ab5f131cf.png width=50% height=50%>

Generate recommendations for a user in the data set:\
python recommend_for_user.py

Plot the 2D latent feature representation for movies:\
python plot_item_features.py\
<img src=https://user-images.githubusercontent.com/103119572/168557119-85ac3f49-0979-44df-8d5d-b9386ee43714.png width=50% height=50%>

## Test scripts

The integrity of the data preprocessing and loading into the custom data structure can be verified with:\
./test/test_datastruct_integrity.py

A sanity check for recommendations produced by the trained model can use:\
python ./test/test_recommendation.py\
Sample output:\
./test/test_recommendation_output.txt

## References

Koenigstein, N., Nice, N., Paquet, U. and Schleyen, N., 2012, September. The Xbox recommender system. In Proceedings of the sixth ACM conference on Recommender systems (pp. 281-284).

Koren, Y., Bell, R. and Volinsky, C., 2009. Matrix factorization techniques for recommender systems. Computer, 42(8), pp.30-37.


