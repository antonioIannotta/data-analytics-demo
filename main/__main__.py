import numpy as np
import pandas as pd
import sys
from utils import movielens_utils

print("Data Analytics Demo")

movie_dataframe = pd.read_csv(sys.argv[1])
tag_relevance_dataframe = pd.read_csv(sys.argv[2])
tag_name_dataframe = pd.read_csv(sys.argv[3])

print("Movie: \n")
print(movie_dataframe)
print("\n")

print("Relevance: \n")
print(tag_relevance_dataframe)
print("\n")

#movie_splitted_genre = movielens_utils.movie_with_splitted_genre(movie_dataframe)
#tag_relevance_movies = movielens_utils.tag_relevance_movies_creation(tag_name_dataframe, tag_relevance_dataframe)


