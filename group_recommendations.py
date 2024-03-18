import pandas as pd
from user_recommendations import make_user_predictions, read_dataset, create_recom_table, P, T, F

U = [1, 610, 207] # Selected USERS (ids) for group recommendations

# Read the datsets and create dataframes
ratings_df = read_dataset(P, T[0])
movies_df = read_dataset(P, T[1])

# Create reccomandations dataframe and do predictions on it
recom_df = create_recom_table(ratings_df)

users_recoms = []
for u in U:
    print(f'\n---------------------- USER {u} ----------------------')
    movie_suggestions = make_user_predictions(u, F[0], recom_df, movies_df)
    users_recoms += movie_suggestions

group_recom = pd.DataFrame(users_recoms)
print(f'\n{group_recom}')

ava_agg = group_recom.groupby('movie')['value'].agg(lambda x: sum(x)/3)
ava_agg_sorted = ava_agg.sort_values(ascending=False)
print(f'\n{ava_agg_sorted}')

min_agg = group_recom.groupby('movie')['value'].agg(lambda x: min(x)* (len(x)/3) if len(x) == 1 or len(x) == 3 else min(x)/2)
min_agg_sorted = min_agg.sort_values(ascending=False)
print(f'\n{min_agg_sorted}')