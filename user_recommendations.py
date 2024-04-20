import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

P = './ml-latest-small/' # Directory PATH of tables
T = ['ratings', 'movies', 'tags', 'links'] # Names of existent TABLES
U = 610 # Selected USER (id) for recommendations
SF = ['pearson', 'custom cosine', 'custom pncr-adf'] # SIMILARITY FUNCTIONS used

def dataset_to_dfs():
    # Read the datsets and create dataframes
    ratings_df = read_dataset(P, T[0])
    movies_df = read_dataset(P, T[1])

    # Create reccomandations dataframe and do predictions on it
    recom_df = create_recom_table(ratings_df)
    
    return recom_df, movies_df

def read_dataset(local_path: str, table: str):
    # Import the selected csv file (table) into dataframe and print its first 10 rows
    df = pd.read_csv(local_path + table + '.csv')
    # print(f'\nTable: {table}')
    # print(df.head(15))
    # print('\nRows, Columns:')
    # print(df.shape)

    # # More infos from "ratings" table
    # if table == 'ratings':
    #     print('Max UserId, Max MovieId:')
    #     print(str(df['userId'].max()) + ', ' + str(df['movieId'].max()))

    return df

def create_recom_table(ratings_df: pd.DataFrame):
    # Create the "user-items rating" dataframe from "ratings" table, used for make predictions
    df = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
    df.dropna(how='all', inplace=True)
    df['mean'] = df.mean(axis=1) # Add mean value of user ratings
    
    # Print first 15 rows and infos
    print('\nTable: user-items rating')
    print(df.head(15))
    print('\nRows, Columns:')
    print(df.shape)
    print('Data sparsity (missing values):')
    print(df.isna().sum().sum() / df.size)
    df.to_csv('./user-items_rating.csv') # Save the new dataframe

    return df

def pearson_similarity(user_a: pd.Series, user_b: pd.Series, movies: pd.Index):
    r_a = user_a[movies] - user_a['mean']
    r_b = user_b[movies] - user_b['mean']

    num = ((r_a) * (r_b)).sum()
    den_a = (r_a ** 2).sum()
    den_b = (r_b ** 2).sum()
    
    den = math.sqrt(den_a * den_b)
    if den == 0:
        return -1
    
    return num/den

def custom_similarity1(user_a: pd.Series, user_b: pd.Series, movies_ids: pd.Index, movies_df: pd.DataFrame):
    # Calculate the Percentage of Non Common Ratings (pncr), which considers the number of co-rated items of the two users
    pncr = np.exp(- ((movies_df.shape[0] - len(movies_ids))/movies_df.shape[0]))

    # Get weights of user_a movies genres for similarity from user_a/user_b vector and "movies" dataframe
    user_a_df = user_a.to_frame('rating').join(movies_df.set_index('movieId')['genres'])
    genres_a = user_a_df['genres'].str.get_dummies()
    weights_a = genres_a.T.dot(user_a_df['rating']) / genres_a.sum()
    
    user_b_df = user_b.to_frame('rating').join(movies_df.set_index('movieId')['genres'])
    genres_b = user_b_df['genres'].str.get_dummies()
    weights_b = genres_b.T.dot(user_b_df['rating']) / genres_b.sum()

    # Get weighted user_a/user_b vector and normalize it
    weighted_ratings_a = user_a.loc[movies_ids] * genres_a.loc[movies_ids].dot(weights_a)
    normalized_ratings_a = normalize(weighted_ratings_a.values.reshape(1, -1))

    weighted_ratings_b = user_b.loc[movies_ids] * genres_b.loc[movies_ids].dot(weights_b)
    normalized_ratings_b = normalize(weighted_ratings_b.values.reshape(1, -1))

    # Get cosine similarity between normalized vectors
    sim = cosine_similarity(normalized_ratings_a, normalized_ratings_b)[0][0]
    return sim * pncr

def custom_similarity2(filt_user_a: pd.Series, filt_user_b: pd.Series, movies_ids: pd.Index, n_movies: int):
    # Compute similiraty based on Percentage of Non Common Ratings (pncr), which considers the number of co-rated items of the two users...
    pncr = np.exp(- ((n_movies - len(movies_ids))/n_movies))

    # ...and Absolute Difference of Ratings (adf), which considers differences in users' ratings
    num_x = (filt_user_a - filt_user_b).abs()
    num_y = np.maximum(filt_user_a, filt_user_b)
    num = np.exp(- (num_x / num_y))
    adf = num.sum()
    
    adf /= len(movies_ids)
    
    # Return the product of the two measures
    return pncr * adf

def get_similar_users(user_row: pd.Series, user_id: int, recom_df: pd.DataFrame, function: str, movies_df: pd.DataFrame):
    df = recom_df.copy()
    df = df.drop(labels = user_id, axis = 0) # Remove selected user from dataframe
    users_sim = []

    # Consider only selected user's existing rating values and print them
    user_a = user_row.dropna() 
    print(f'\nSelected User: {user_a.name}')
    
    # Get similarities between selected user and all other users on common movies, using function passed as parameter
    if function == SF[0]:
        for user_b_id in df.index:
            user_b = df.loc[user_b_id].dropna()
            movies = user_a.index.intersection(user_b.index)
            users_sim.append({'id': user_b_id, 'sim': pearson_similarity(user_a, user_b, movies)})
    elif function == SF[1]:
        for user_b_id in df.index:
            user_b = df.loc[user_b_id].dropna()
            movies = user_a.index.intersection(user_b.index)
            users_sim.append({'id': user_b_id, 'sim': custom_similarity1(user_a, user_b, movies, movies_df)}) # Custom Cosine Similarity
    elif function == SF[2]:
        for user_b_id in df.index:
            user_b = df.loc[user_b_id].dropna()
            movies = user_a.index.intersection(user_b.index)
            users_sim.append({'id': user_b_id, 'sim': custom_similarity2(user_a[movies], user_b[movies], movies, movies_df.shape[0])}) # Custom pncr-adf similarity
    else:
        print('There is no function with selected name')
        return
    
    # Sort user similarities list in descending order
    users_sim_sorted = sorted(users_sim, key=lambda x: x['sim'], reverse=True)
    
    print(f'\nSimilarities ({function}): ')
    for user in users_sim_sorted[:10]:
        print(user)
    
    # Return the first 50 similar users
    users_sim_filt = users_sim_sorted[:50]
    return users_sim_filt

def prediction_function(user_a: pd.Series, users: list, movie: int, recom_df: pd.DataFrame):
    # Calculate predictions with a common function
    r_mean_a = recom_df.at[user_a, 'mean']
    num_list = []
    den_list = []

    for u in users:
        r_m_b = recom_df.at[u['id'], movie]
        if not pd.isna(r_m_b):
            sim = u['sim']
            r_mean_b = recom_df.at[u['id'], 'mean']
            r_b = r_m_b - r_mean_b
            num_list.append(sim * r_b)
            den_list.append(sim)
    
    num = np.sum(num_list)
    den = np.sum(den_list)

    if den == 0:
        return r_mean_a
    
    div = num/den
    return r_mean_a + div


def get_recommendations(user_id: int, user: pd.Series, sim_users: list, recom_df: pd.DataFrame):
    # Get the unseen movies of selected user
    unseen_movies = user[user.isna()].index

    # Make prediction for all selected user's unseen movies
    suggestions = [{'movie': movie, 'value': prediction_function(user_id, sim_users, movie, recom_df)} for movie in unseen_movies]
    suggestions_sorted = sorted(suggestions, key=lambda x: x['value'], reverse=True)
    
    for sug in suggestions_sorted[:10]:
        print(sug)

    return suggestions_sorted

def make_user_predictions(user_id: int, function: str, recom_df: pd.DataFrame, movies_df: pd.DataFrame):
    # Extract the user on which to make predictions
    user_row = recom_df.loc[user_id]

    if function == SF[0]:
        sim_users = get_similar_users(user_row, user_id, recom_df, SF[0], movies_df)
        print(f'\nRecommendations ({SF[0]}): ')
        movie_suggestions = get_recommendations(user_id, user_row, sim_users, recom_df)
    elif function == SF[1]:
        sim_users = get_similar_users(user_row, user_id, recom_df, SF[1], movies_df)
        print(f'\nRecommendations ({SF[1]}): ')
        movie_suggestions = get_recommendations(user_id, user_row, sim_users, recom_df)
    elif function == SF[2]:
        sim_users = get_similar_users(user_row, user_id, recom_df, SF[2], movies_df)
        print(f'\nRecommendations ({SF[2]}): ')
        movie_suggestions = get_recommendations(user_id, user_row, sim_users, recom_df)
    else:
        print('There is no function with selected name')
        return

    return movie_suggestions

def main_user():
    # Convert dataset csv in dataframes
    recom_df, movies_df = dataset_to_dfs()
    
    # Execute user recommendations using both Pearson and the custom similarity functions
    user_recom_pearson = make_user_predictions(U, SF[0], recom_df, movies_df)
    user_recom_custom = make_user_predictions(U, SF[1], recom_df, movies_df)

if __name__ == '__main__':
    main_user()