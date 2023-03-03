import pandas as pd
import numpy as np


def movie_with_splitted_genre(movie):
    genres = []
    for i in range(len(movie.genres)):
        for x in movie.genres[i].split('|'):
            if x not in genres:
                genres.append(x)

    len(genres)
    for x in genres:
        movie[x] = 0
    for i in range(len(movie.genres)):
        for x in movie.genres[i].split('|'):
            movie[x][i] = 1

    return movie


def tag_relevance_movies_creation(tag_name_dataframe, tag_relevance_dataframe):
    movieIds = tag_relevance_dataframe.groupby('movieId')
    tag_columns = []
    for i in range(len(tag_name_dataframe)):
        tag_columns.append(tag_name_dataframe.iloc[i, 0])

    tag_relevance_movies = pd.DataFrame({
        'movieId': movieIds.groups.keys()
    })

    y = len(tag_relevance_movies.columns)
    for i in range(len(tag_columns)):
        tag_relevance_movies.insert(y, tag_columns[i], "")
        y += 1

    for i in range(len(tag_relevance_movies.movieId)):
        x = movieIds.get_group(tag_relevance_movies.iloc[i, 0])
        z = np.array(x.relevance)
        tag_relevance_movies.iloc[i, 1:] = z[:]

    return tag_relevance_movies
