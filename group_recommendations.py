import pandas as pd
from user_recommendations import make_user_predictions, dataset_to_dfs

U = [1, 610, 207] # Selected USERS (ids) for group recommendations
PR = 0 # The "PENALTY RATING" replacing ratings of movies not recommended to all group's users in avarage Aggregation
AM = ['ava', 'lm'] # AGGREGATION METHODS: Avarage and Least Misery

def make_group_predictions(recom_df: pd.DataFrame, movies_df: pd.DataFrame, users: list[int]):
    individual_recoms: list[pd.DataFrame] = []
    
    for u in users:
        print(f'\n---------------------- USER {u} ----------------------')
        user_predictions = make_user_predictions(u, 'custom', recom_df, movies_df)
        user_predictions_df = pd.DataFrame(user_predictions).set_index('movie')
        normalized_predictions_df = normalize_recoms(user_predictions_df, 0.5, 5.0)
        print(normalized_predictions_df)
        individual_recoms.append(normalized_predictions_df)
    
    group_recom = pd.concat(individual_recoms, axis=1)
    group_recom.columns = [f'{u}' for u in users]
    group_recom.dropna(how='all', inplace=True)
    
    # Fill NaN values with real users' ratings
    recom_df_T = recom_df.loc[users].transpose()
    recom_df_T.columns = group_recom.columns
    group_recom.fillna(recom_df_T, inplace=True)
    
    print('\nUsers Recommendations')
    print(f'\n{group_recom}')
    print('\nRecom describe')
    print(group_recom.describe(include='all'))

    return individual_recoms, group_recom

def normalize_recoms(recoms: pd.Series, min_value: float, max_value: float):
    min_rating = recoms.min()
    max_rating = recoms.max()
    
    normalized_ratings = (recoms - min_rating) / (max_rating - min_rating) * (max_value - min_value) + min_value
    return normalized_ratings

def get_disagreements(group_recom: pd.DataFrame):
    group_recom_dis = pd.DataFrame(index=group_recom.index)
    group_recom_dis['std'] = group_recom.std(axis=1, skipna=True)
    # group_recom_dis['range'] = group_recom.max(axis=1) - group_recom.min(axis=1)
    # group_recom_dis['var'] = group_recom.var(axis=1)
    
    print('\nUsers Disagreements: ')
    print(group_recom_dis)

    inverted_std = 1 / group_recom_dis['std']

    weights = inverted_std / inverted_std.sum()
    weighted_group_recom = group_recom.mul(weights, axis=0)

    return weighted_group_recom

def get_aggregation_and_fairness(weighted_group_recom: pd.DataFrame, recoms: list[pd.DataFrame], agg_methods: list[str], dis: bool):
    if not dis:
        print('\n-------------- AVARAGE AGGREGATION --------------')
        ava_agg_recom = aggregation(weighted_group_recom, agg_methods[0])
        
        print('\nAll:\n')
        ava_group_fairness = fairness(recoms, ava_agg_recom, True)
        print('Group Fairness:')
        print(f'{ava_group_fairness}\n')
        
        print('\nTop 100:\n')
        ava_group_fairness = fairness(recoms, ava_agg_recom, False, 100)
        print('Group Fairness:')
        print(f'{ava_group_fairness}\n')
        
        print('\n-------------- LEAST MISERY AGGREGATION --------------')
        lm_agg_recom = aggregation(weighted_group_recom, agg_methods[1])
        
        print('\nAll:\n')
        lm_group_fairness = fairness(recoms, lm_agg_recom, True)
        print('Group Fairness:')
        print(f'{lm_group_fairness}\n')
        
        print('\nTop 100:\n')
        lm_group_fairness = fairness(recoms, lm_agg_recom, False, 100)
        print('Group Fairness:')
        print(f'{lm_group_fairness}\n')

    if dis:
        print('\n-------------- UPWARD LEVELING AGGREGATION --------------')
        ul_agg_recom = ul_aggregation(weighted_group_recom, [0.4, 0.2, 0.4])
        
        print('\nAll:\n')
        ul_group_fairness = fairness(recoms, ul_agg_recom, True)
        print('Group Fairness:')
        print(f'{ul_group_fairness}\n')
        
        print('\nTop 100:\n')
        ul_group_fairness = fairness(recoms, ul_agg_recom, False, 100)
        print('Group Fairness:')
        print(f'{ul_group_fairness}\n')

def aggregation(weighted_group_recom: pd.DataFrame, agg_method: list[str]):
    df = weighted_group_recom.copy()
    
    if agg_method == 'ava':
        df.fillna(PR, inplace=True)
        agg_recom = df.mean(axis=1)
    elif agg_method == 'lm':
        agg_recom = df.agg(lambda x: min(x) * (x.dropna().shape[0]/3) if x.dropna().shape[0] == 1 or x.dropna().shape[0] == 3 else min(x)/2, axis=1)
    
    agg_recom.sort_values(ascending=False, inplace=True)
    print('\nAggregation list: ')
    print(agg_recom.head(10))

    return agg_recom

def ul_aggregation(group_recom: pd.DataFrame, coeff: list[float]):
    df = group_recom.copy()
    ul_score = 0

    mean_recom = df.mean(axis=1)
    max_rating_recom = df.agg(lambda x: x[x > 3.5].count(), axis=1)
    
    # Calculate the mean of the squared deviations
    squared_deviations = pd.DataFrame()
    for col in df.columns:
        squared_deviations[col] = mean_recom - df[col]
    squared_deviations = squared_deviations ** 2
    msd_recom = squared_deviations.mean(axis=1)
    dev_recom = 1 - msd_recom

    ul_score = (mean_recom * coeff[0]) + (max_rating_recom * coeff[1]) + (dev_recom * coeff[2])
    ul_score.sort_values(ascending=False, inplace=True)
    
    print('\nUL SCORE:')
    print(ul_score.head(10))
    return ul_score

def fairness(individual_recoms: list[pd.DataFrame], agg_recom: pd.Series, all_list: bool, top: int = None):
    individual_r = individual_recoms.copy()
    agg_r = agg_recom.copy()
    group_fairness = 0
    
    if not all_list:
        agg_r = agg_r.head(top)
    
    for i in range(len(individual_r)):
        if not all_list:
            individual_r[i] = individual_r[i].head(top)
        
        x = individual_r[i].index.intersection(agg_r.index)
        user_fairness = x.shape[0] / agg_r.shape[0]
        print(f'User {U[i]} fairness:')
        print(user_fairness)
        
        group_fairness += user_fairness

    return group_fairness / len(individual_r)

if __name__ == '__main__':
    recom_df, movies_df = dataset_to_dfs()
    
    # Execute users' individual recommendations and merdge them in one dataframe
    individual_recoms, group_recom = make_group_predictions(recom_df, movies_df, U)
    
    print('\n---------------------- NO DISAGREEMENT ----------------------')
    get_aggregation_and_fairness(group_recom, individual_recoms, AM, False)

    weighted_group_recom = get_disagreements(group_recom)

    print('\n---------------------- WITH DISAGREEMENT ----------------------')
    get_aggregation_and_fairness(weighted_group_recom, individual_recoms, AM, True)

