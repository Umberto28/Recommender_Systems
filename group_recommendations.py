import pandas as pd
from user_recommendations import make_user_predictions, dataset_to_dfs, SF

U = [1, 610, 207] # Selected USERS (ids) for group recommendations
PR = 0 # The "PENALTY RATING" replacing ratings of movies not recommended to all group's users in avarage Aggregation
AM = ['avarage', 'least misery'] # AGGREGATION METHODS: Avarage and Least Misery

def make_group_predictions(recom_df: pd.DataFrame, movies_df: pd.DataFrame, users: list[int], sim_method: str):
    individual_recoms: list[pd.DataFrame] = []
    
    # Make prediction for each user in the group and normalize scores
    for u in users:
        print(f'\n---------------------- USER {u} ----------------------')
        user_predictions = make_user_predictions(u, sim_method, recom_df, movies_df)
        user_predictions_df = pd.DataFrame(user_predictions).set_index('movie')
        normalized_predictions_df = normalize_recoms(user_predictions_df, 0.5, 5.0)
        individual_recoms.append(normalized_predictions_df)
    
    # Merging all individual prediction in a single dataframe
    group_recom = pd.concat(individual_recoms, axis=1)
    group_recom.columns = [f'{u}' for u in users]
    group_recom.dropna(how='all', inplace=True)
    
    # Fill NaN values with real users' ratings
    recom_df_T = recom_df.loc[users].transpose()
    recom_df_T.columns = group_recom.columns
    group_recom.fillna(recom_df_T, inplace=True)
    
    print('\nUsers Recommendations')
    print(f'\n{group_recom}')
    # print('\nRecom describe')
    # print(group_recom.describe(include='all'))

    return individual_recoms, group_recom

def normalize_recoms(recoms: pd.Series, min_value: float, max_value: float):
    # Normalize recommendation scores between 'min_value' and 'max_value'
    min_rating = recoms.min()
    max_rating = recoms.max()
    
    normalized_ratings = (recoms - min_rating) / (max_rating - min_rating) * (max_value - min_value) + min_value
    return normalized_ratings

def get_aggregation_and_fairness(group_recom: pd.DataFrame, recoms: list[pd.DataFrame], agg_methods: list[str], dis: bool):
    # Get group recommendation aggregation considering or not users' disagreement (choosen using 'dis' parameter)
    if dis:
        print('\n-------------- UPWARD LEVELING AGGREGATION --------------')
        ul_agg_recom = ul_aggregation(group_recom, [0.35, 0.2, 0.45])

        # print('\nAll:\n')
        # ul_group_fairness = fairness(recoms, ul_agg_recom, True)
        # print('Group Fairness:')
        # print(f'{ul_group_fairness}\n')
        
        print('\nTop 100:\n')
        ul_group_fairness = fairness(recoms, ul_agg_recom, False, 100)
        print('Group Fairness:')
        print(f'{ul_group_fairness}\n')
    
    else:
        for a in agg_methods:
            print(f'\n-------------- {a.upper()} AGGREGATION --------------')
            agg_recom = standard_aggregation(group_recom, a)
            
            # print('\nAll:\n')
            # group_fairness = fairness(recoms, agg_recom, True)
            # print('Group Fairness:')
            # print(f'{group_fairness}\n')
            
            print('\nTop 100:\n')
            group_fairness = fairness(recoms, agg_recom, False, 100)
            print('Group Fairness:')
            print(f'{group_fairness}\n')

def standard_aggregation(group_recom: pd.DataFrame, agg_method: list[str]):
    # Compute both Avarage and Least Misery aggregation methods to group recommendations
    df = group_recom.copy()
    
    if agg_method == 'avarage':
        df.fillna(PR, inplace=True)
        agg_recom = df.mean(axis=1)
    elif agg_method == 'least misery':
        agg_recom = df.agg(lambda x: min(x) * (x.dropna().shape[0]/3) if x.dropna().shape[0] == 1 or x.dropna().shape[0] == 3 else min(x)/2, axis=1)
    else:
        print('There is no function with selected name')
        return
    
    agg_recom.sort_values(ascending=False, inplace=True)
    print('\nAggregation list: ')
    print(agg_recom.head(10))

    return agg_recom

def ul_aggregation(group_recom: pd.DataFrame, coeff: list[float]):
    # Compute the Upward Leveling aggregation method, a combined method that use Avarage, Approval Voting and Mean Square Deviation with coefficients
    df = group_recom.copy()
    ul_score = 0

    # Calculate avarage and approval voting
    mean_recom = df.mean(axis=1)
    max_rating_recom = df.agg(lambda x: x[x > 3.9].count(), axis=1)
    
    # Calculate the mean squared deviations
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
    # Calculate user fairness as division between number of user's recommended movies with 'good' rating (over 3.9) intersectioned with number of all group recommended movies and number of all group recommended movies
    individual_r = individual_recoms.copy()
    agg_r = agg_recom.copy()
    group_fairness = 0
    
    # Check if you want to consider all list or just the first 'top' items
    if not all_list:
        agg_r = agg_r.head(top)
    
    for i in range(len(individual_r)):
        if not all_list:
            individual_r[i] = individual_r[i][individual_r[i]['value'] >= 3.9] 
        x = individual_r[i].index.intersection(agg_r.index)
        user_fairness = x.shape[0] / agg_r.shape[0]
        print(f'User {U[i]} fairness:')
        print(user_fairness)
        
        group_fairness += user_fairness

    return group_fairness / len(individual_r)

def main_group():
    # Convert dataset csv in dataframes
    recom_df, movies_df = dataset_to_dfs()
    
    # Execute users' individual recommendations and merge them in one dataframe
    individual_recoms, group_recom = make_group_predictions(recom_df, movies_df, U, SF[1])
    
    # Aggregate results using standard aggregation methods (which don't consider disagreement) and get fairness
    print('\n---------------------- NO DISAGREEMENT ----------------------')
    get_aggregation_and_fairness(group_recom, individual_recoms, AM, False)

    # Aggregate results using another aggregation method (which considers disagreement) and get fairness
    print('\n---------------------- WITH DISAGREEMENT ----------------------')
    get_aggregation_and_fairness(group_recom, individual_recoms, AM, True)

if __name__ == '__main__':
    main_group()

