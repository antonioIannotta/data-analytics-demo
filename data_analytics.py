import numpy as np
import pandas as pd
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import movielens_utils
import pickle as pk

print("Data Analytics Demo")

path = "./csv_files_input/" + sys.argv[1]
movie_dataframe = pd.read_csv(path)
path = "./csv_files_input/" + sys.argv[2]
tag_relevance_dataframe = pd.read_csv(path)
path = "./csv_files_input/" + sys.argv[3]
tag_name_dataframe = pd.read_csv(path)

print("*****************DATASET********************\n")

print("Movie: \n")
print(movie_dataframe)
print("\n")

print("Tag Relevance: \n")
print(tag_relevance_dataframe)
print("\n")

print("Tag Name: \n")
print(tag_name_dataframe)
print("\n")

movie_splitted_genre = movielens_utils.movie_with_splitted_genre(movie_dataframe)
tag_relevance_movies = movielens_utils.tag_relevance_movies_creation(tag_name_dataframe, tag_relevance_dataframe)

final_dataframe = pd.merge(movie_splitted_genre, tag_relevance_movies, on='movieId')
final_dataframe = final_dataframe.drop(columns='movieId')

print("****************** FINAL DATASET **************\n")
print(final_dataframe)
print("\n")


print("NON-SUPERVISED METHODS")
print("Application of PCA")

standard_scaler = StandardScaler()
final_dataframe_scale = standard_scaler.fit_transform(final_dataframe)

pca = PCA(n_components=0.70)
final_df_pca = pca.fit_transform(final_dataframe_scale)

knn_regression = pk.load('nd_supervised_models/knn_regression.pkl')
knn_predicted = knn_regression.predict(final_df_pca)



