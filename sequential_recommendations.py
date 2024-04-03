import pandas as pd
from user_recommendations import dataset_to_dfs, get_similar_users, SF
from group_recommendations import make_group_predictions, ul_aggregation, U

UID = 610 # From this USER ID select the first two most similar users to create the group
I = 3 # Number of ITERATIONS
UL_C = [0.35, 0.2, 0.45] # COEFFICIENTS for the UL_Aggregation formula
T = 100 # The TOP recommended items to consider calculating satisfaction

def get_group_users(random: bool, recom_df: pd.DataFrame, movies_df: pd.DataFrame, user_id: int):
    # Select three users to put in the group for sequential recommendation 
    group = []

    # They can be choosen randomly...
    if random:
        group = U
    # ...or taking one random user and then select the first two most similar users
    else:
        group.append(user_id)
        sim_users = get_similar_users(recom_df.loc[user_id], user_id, recom_df, SF[2], movies_df)
        group.append(sim_users[0]['id'])
        group.append(sim_users[1]['id'])

    return group

def satisfaction(group_recoms: pd.DataFrame, agg_recom: pd.Series, users: list[int], top: int, iteration: int, individual_sat: pd.DataFrame):
    # Calculate users' individal satisfaction based on scores in the group recommendation list
    agg_recom_top = agg_recom.head(top)

    # Individual satisfaction is get dividing sum of aggregated recommendation scores by sum of user recommendation scores
    for i in range(len(users)):
        i_recoms = group_recoms[str(users[i])].head(top)
        group_list_sat = group_recoms.loc[agg_recom_top.index, str(users[i])].sum()
        user_list_sat = group_recoms.loc[i_recoms.index, str(users[i])].sum()
        sat = group_list_sat/user_list_sat
        individual_sat.loc[i] = [int(users[i]), sat, 0]
        individual_sat.at[i, 'satO_num'] += individual_sat.at[i, 'sat']
    
    # Calculate overall individual satisfaction, group satisfaction (and overall), alpha for the next iteation and group disagreement
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
    
    return individual_sat, alpha

if __name__ == '__main__':
    # Convert dataset csv in dataframes
    recom_df, movies_df = dataset_to_dfs()

    # Create the group of users for the sequential recommendation
    group = get_group_users(True, recom_df, movies_df, UID)

    # Compute group recommendation and get recommended lists
    individual_recoms, group_recom = make_group_predictions(recom_df, movies_df, group, SF[2])

    # Initialize variables to use in each iteration of sequential recommendation and compute it
    alpha = 0
    individual_sat = pd.DataFrame(index=[0, 1, 2], columns=['user', 'sat', 'satO_num'])
    for i in range(I):
        print(f'\n---------------------- ITERATION {i+1} ----------------------')
        # Aggregate group recommendation weighted with alpha (a form of disagreement)
        agg_recom = ul_aggregation(group_recom, UL_C) * (1 - alpha)
        
        # Compute users' satisfaction and use it s weights for inidividual recommendations
        individual_sat, alpha = satisfaction(group_recom, agg_recom, group, T, i+1, individual_sat)
        for j in range(len(group)):
            group_recom[group_recom.columns[j]] *= (1 - individual_sat.at[j, 'sat'])