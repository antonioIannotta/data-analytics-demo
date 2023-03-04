import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from utils import movielens_utils
import pickle as pk
import torch

print("Data Analytics Demo")

path_movie = "./csv_files_input/" + sys.argv[1]
movie_dataframe = pd.read_csv(path_movie)
path_genome_scores = "./csv_files_input/" + sys.argv[2]
tag_relevance_dataframe = pd.read_csv(path_genome_scores)
path_genome_tag = "./csv_files_input/" + sys.argv[3]
tag_name_dataframe = pd.read_csv(path_genome_tag)

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


print("*********************+NON-DEEP METHODS*******************************")
final_dataframe.columns = final_dataframe.columns.astype('str')
X = final_dataframe.iloc[:, :]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print("Shape after scaling: " + str(X.shape))
print("Scaled dataset")
print(X)

pca = PCA(n_components=0.70)
X_t = pca.fit_transform(X)

print("PCA result")
print(X_t.shape)

knn_regression = pk.load(open('nd_supervised_models/knn_regression.pkl', "rb"))
knn_predicted = knn_regression.predict(X_t)
print("Prediction with KNN: " + str(knn_predicted))

linear_regression = pk.load(open(open('nd_supervised_models/linear_regression.pkl', "rb")))
linear_predicted = linear_regression.predict(X_t)
print("Prediction with Linear regression: " + str(linear_predicted))

random_forest_regression = pk.load(open(open('nd_supervised_models/random_forest_regression.pkl', "rb")))
random_predicted = random_forest_regression.predict(X_t)
print("Prediction with Random forest: " + str(random_predicted))


print("*******************************************DEEP LEARNING*****************************************")


dropout_model = torch.load('./neaural_network/best_model_dropout/data.pkl')
no_dropout_model = torch.load('./neaural_network/best_model_without_dropout/')

dl_pca = PCA(n_components=0.95)
X_dl = dl_pca.fit_transform(X)

dropout_pred = dropout_model.predict(dl_pca)
print("Prediction with dropout: " + str(dropout_pred))

no_dropout_pred = dropout_model.predict(dl_pca)
print("Prediction without dropout: " + str(no_dropout_pred))

print("******************************************TABNET*************************************************")

tabnet_model = torch.load('./tabnet/model/data.pkl')
tabnet_pred = tabnet_model.predict(X)

print("Prediction with tabnet model: " + str(tabnet_pred))


