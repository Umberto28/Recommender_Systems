import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

P = './ml-latest-small/' # Directory PATH of tables
T = ['ratings', 'movies', 'tags', 'links'] # Names of existent TABLES
U = 610 # Selected USER (id) for recommendations
F = ['pearson', 'custom'] # Similarity FUNCTIONS used

def read_dataset(local_path, table):
    # Import the selected csv file (table) into dataframe and print its first 10 rows
    df = pd.read_csv(local_path + table + '.csv')
    print(f'\nTable: {table}')
    print(df.head(10))
    print('\nRows, Columns:')
    print(df.shape)

    # More infos from "ratings" table
    if table == 'ratings':
        print('Max UserId, Max MovieId:')
        print(str(df['userId'].max()) + ', ' + str(df['movieId'].max()))

    return df

def create_recom_table(ratings_df):
    # Create the "user-items rating" dataframe from "ratings" table, used for make predictions
    df = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
    df['mean'] = df.mean(axis=1) # Add mean value of user ratings
    
    # Print first 15 rows and infos
    print('\nTable: user-items rating')
    print(df.head(15))
    print('\nRows, Columns:')
    print(df.shape)
    df.to_csv('./user-items_rating.csv') # Save the new dataframe

    return df

def pearson_similarity(user_a, user_b, movies):
    num = 0
    den_a = 0
    den_b = 0
    
    r_mean_a = user_a['mean']
    r_mean_b = user_b['mean']

    for m in movies:
        r_a = user_a[m] - r_mean_a
        r_b = user_b[m] - r_mean_b
        num += (r_a) * (r_b)
        den_a += math.pow(r_a, 2)
        den_b += math.pow(r_b, 2)
    
    den = math.sqrt(den_a * den_b)
    if den == 0:
        print(f'There\'s not correlation between user {user_a.name} and user {user_b.name}')
        return -1
    
    return num/den

def custom_cosine_similarity(user_a, user_b, movies_ids, movies_df):
    df = movies_df.copy()
    # Get weights of user_a movies genres for similarity from user_a vector and "movies" dataframe
    user_a_df = user_a.to_frame('rating').join(df.set_index('movieId')['genres'])
    genres_a = user_a_df['genres'].str.get_dummies()
    weights_a = genres_a.T.dot(user_a_df['rating']) / genres_a.sum()
    
    # Get weights of user_b movies genres for similarity from user_b vector and "movies" dataframe
    user_b_df = user_b.to_frame('rating').join(df.set_index('movieId')['genres'])
    genres_b = user_b_df['genres'].str.get_dummies()
    weights_b = genres_b.T.dot(user_b_df['rating']) / genres_b.sum()

    # Get weighted users vectors and normalize them
    weighted_ratings_a = user_a.loc[movies_ids] * genres_a.loc[movies_ids].dot(weights_a)
    weighted_ratings_b = user_b.loc[movies_ids] * genres_b.loc[movies_ids].dot(weights_b)
    normalized_ratings_a = normalize(weighted_ratings_a.values.reshape(1, -1))
    normalized_ratings_b = normalize(weighted_ratings_b.values.reshape(1, -1))

    # Get cosine similarity between normalized vectors
    sim = cosine_similarity(normalized_ratings_a, normalized_ratings_b)[0][0]
    norm_sim = sim/np.log1p(len(movies_ids))
    return norm_sim

def prediction_function(user_a, users, movie, recom_df):
    df = recom_df.copy()
    r_mean_a = df.at[user_a, 'mean']
    num = 0
    den = 0

    for u in users:
        r_m_b = df.at[u['id'], movie]
        if not pd.isna(r_m_b):
            sim = u['sim']
            r_mean_b = df.at[u['id'], 'mean']
            r_b = r_m_b - r_mean_b
            num += sim * r_b
            den += sim
    
    if den == 0:
        return r_mean_a
    
    div = num/den
    return r_mean_a + div

def get_similar_users(user_row, user_id, recom_df, function, movies_df):
    df = recom_df.copy()
    df = df.drop(labels = user_id, axis = 0) # Remove selected user from dataframe
    users_sim = []

    # Consider only selected user's existing rating values and print them
    user_a = user_row.dropna() 
    print('\nSelected User: ')
    print(user_a)
    
    # Get similarities between selected user and all other users on common movies, using function passed as parameter
    if function == F[0]:
        for user_b_id in df.index:
            user_b = df.loc[user_b_id].dropna()
            movies = user_a.index.intersection(user_b.index)
            sim = pearson_similarity(user_a, user_b, movies)
            users_sim.append({'id': user_b_id, 'sim': sim})
    elif function == F[1]:
        for user_b_id in df.index:
            user_b = df.loc[user_b_id].dropna()
            movies = user_a.index.intersection(user_b.index)
            sim = custom_cosine_similarity(user_a, user_b, movies, movies_df)
            users_sim.append({'id': user_b_id, 'sim': sim})
    
    # Sort user similarities list in descending order
    users_sim_sorted = sorted(users_sim, key=lambda x: x['sim'], reverse=True)
    
    print(f'\nSimilarities ({function}): ')
    for user in users_sim_sorted[:11]:
        print(user)
    
    return users_sim[:50]

def get_recommendations(user_id, user, sim_users, recom_df):
    suggestions = []

    for movie in user.index:
        if pd.isna(user[movie]):
            value = prediction_function(user_id, sim_users, movie, recom_df)
            suggestions.append({'movie': movie, 'value': value})
    suggestions_sorted = sorted(suggestions, key=lambda x: x['value'], reverse=True)
    
    for sug in suggestions_sorted[:10]:
        print(sug)

    return suggestions_sorted

def make_user_predictions(user_id, function, recom_df, movies_df):
    df = recom_df.copy()
    # Extract the user on which to make predictions
    user_row = df.loc[user_id]

    if function == F[0]:
        sim_users = get_similar_users(user_row, user_id, recom_df, F[0], movies_df)
        print(f'\nRecommendations ({F[0]}): ')
        movie_suggestions = get_recommendations(user_id, user_row, sim_users, recom_df)
    elif function == F[1]:
        sim_users = get_similar_users(user_row, user_id, recom_df, F[1], movies_df)
        print(f'\nRecommendations ({F[1]}): ')
        movie_suggestions = get_recommendations(user_id, user_row, sim_users, recom_df)

    return movie_suggestions

def main():
    # Read the datsets and create dataframes
    ratings_df = read_dataset(P, T[0])
    movies_df = read_dataset(P, T[1])

    # Create reccomandations dataframe and do predictions on it
    recom_df = create_recom_table(ratings_df)
    user_recom_pearson = make_user_predictions(U, F[0], recom_df, movies_df)
    user_recom_custom = make_user_predictions(U, F[1], recom_df, movies_df)
    return

if __name__ == '__main__':
    main()