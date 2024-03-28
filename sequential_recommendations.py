import pandas as pd
from user_recommendations import dataset_to_dfs, get_similar_users
from group_recommendations import make_group_predictions, normalize_recoms, U

UID = 610
I = 5 # Number of ITERATIONS
UL_C = [0.4, 0.2, 0.4] # COEFFICIENTS for the UL_Aggregation formula
T = 100 # The TOP recommended items to consider calculating satisfaction

def get_group_users(random: bool, recom_df: pd.DataFrame, movies_df: pd.DataFrame, user_id: int):
    group = []

    if random:
        group = U
    else:
        group.append(user_id)
        sim_users = get_similar_users(recom_df.loc[user_id], user_id, recom_df, 'custom', movies_df)
        group.append(sim_users[0]['id'])
        group.append(sim_users[1]['id'])

    return group

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

def satisfaction(group_recoms: pd.DataFrame, agg_recom: pd.Series, users: list[int], top: int, iteration: int, satO_contribute: pd.Series):
    individual_sat = pd.DataFrame(columns=['user', 'sat', 'satO_num'])
    agg_recom_top = agg_recom.head(top)

    for i in range(len(users)):
        i_recoms = group_recoms[str(users[i])].head(top)
        group_list_sat = group_recoms.loc[agg_recom_top.index, str(users[i])].sum()
        user_list_sat = group_recoms.loc[i_recoms.index, str(users[i])].sum()
        sat = group_list_sat/user_list_sat
        individual_sat.loc[i] = [int(users[i]), sat, 0]
        individual_sat.loc[i, 'satO_num'] = satO_contribute.loc[i] + individual_sat.loc[i, 'sat']
    
    users_satO = (individual_sat['satO_num'] / iteration)
    g_sat = individual_sat['sat'].mean()
    g_satO = users_satO.mean()

    alpha = individual_sat['sat'].max() - individual_sat['sat'].min()
    groupDis = users_satO.max() - users_satO.min()

    print('\nGroup Satisfaction:')
    print(g_sat)
    print('Overall Group Satisfaction:')
    print(g_satO)
    print('Group Disagreement:')
    print(groupDis)
    
    return individual_sat['sat'], alpha, individual_sat['satO_num']

if __name__ == '__main__':
    recom_df, movies_df = dataset_to_dfs()

    group = get_group_users(True, recom_df, movies_df, UID)

    individual_recoms, group_recom = make_group_predictions(recom_df, movies_df, group)

    alpha = 0
    satO_contribute = pd.Series([0, 0, 0])
    for i in range(I):
        print(f'\n---------------------- ITERATION {i+1} ----------------------')
        agg_recom = ul_aggregation(group_recom, UL_C) * (1 - alpha)
        individual_sat, alpha, satO_contribute = satisfaction(group_recom, agg_recom, group, T, i+1, satO_contribute)

        for j in range(len(group)):
            group_recom[group_recom.columns[j]] *= (1 - individual_sat[j])