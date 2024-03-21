import pandas as pd
from user_recommendations import make_user_predictions, read_dataset, create_recom_table, P, T, F

U = [1, 610, 207] # Selected USERS (ids) for group recommendations
PR = 0 # The "Penalty Rating" replacing ratings of movies not recommended to all users in the group

def make_group_predictions(recom_df, movies_df):
    users_recoms = []
    for u in U:
        print(f'\n---------------------- USER {u} ----------------------')
        movie_suggestions = make_user_predictions(u, F[0], recom_df, movies_df)
        # movie_suggestions_df = pd.Series([movie['value'] for movie in movie_suggestions], index=[movie['movie'] for movie in movie_suggestions])
        users_recoms.append(movie_suggestions)

    df1 = pd.DataFrame(users_recoms[0]).set_index('movie')
    df2 = pd.DataFrame(users_recoms[1]).set_index('movie')
    df3 = pd.DataFrame(users_recoms[2]).set_index('movie')

    group_recom = pd.concat([df1, df2, df3], axis=1)
    # group_recom = pd.DataFrame({f'user{U[0]}': users_recoms[0], f'user{U[1]}': users_recoms[1], f'user{U[2]}': users_recoms[2]}, )
    group_recom.columns = [f'user{u}' for u in U]
    group_recom.dropna(how='all', inplace=True)
    print('\nUsers Recommendations')
    print(f'\n{group_recom}')
    
    weighted_df = disagreements(group_recom)

    avarage_aggregation(weighted_df)
    least_misery_aggregation(weighted_df)

def disagreements(group_recom):
    group_recom_dis = pd.DataFrame(index=group_recom.index)
    group_recom_dis['std'] = group_recom.std(axis=1, skipna=True)
    # group_recom_dis['range'] = group_recom.max(axis=1) - group_recom.min(axis=1)
    # group_recom_dis['var'] = group_recom.var(axis=1)
    print('\nUsers Disagreements: ')
    print(group_recom_dis)

    weight = 1 / group_recom_dis['std']
    weighted_df = group_recom.mul(weight, axis=0)

    return weighted_df

def avarage_aggregation(weighted_df):
    df = weighted_df.copy()
    df.fillna(PR, inplace=True)
    ava_agg = df.agg(lambda x: sum(x)/3, axis=1)
    ava_agg_sorted = ava_agg.sort_values(ascending=False)
    print('\nAvarage Aggregation: ')
    print(ava_agg_sorted.head(10))

def least_misery_aggregation(weighted_df):
    min_agg = weighted_df.agg(lambda x: min(x) * (x.dropna().shape[0]/3) if x.dropna().shape[0] == 1 or x.dropna().shape[0] == 3 else min(x)/2, axis=1)
    min_agg_sorted = min_agg.sort_values(ascending=False)
    print('\nLeast Misery Aggregation: ')
    print(min_agg_sorted.head(10))

if __name__ == '__main__':
    # Read the datsets and create dataframes
    ratings_df = read_dataset(P, T[0])
    movies_df = read_dataset(P, T[1])

    # Create reccomandations dataframe and do predictions on it
    recom_df = create_recom_table(ratings_df)
    
    make_group_predictions(recom_df, movies_df)

