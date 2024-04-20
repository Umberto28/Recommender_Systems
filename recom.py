import user_recommendations
import group_recommendations
import sequential_recommendations

if __name__ == '__main__':
    print(f'\n-------------------------------- USER RECOM --------------------------------')
    user_recommendations.main_user()
    print(f'\n-------------------------------- GROUP RECOM --------------------------------')
    group_recommendations.main_group()
    print(f'\n-------------------------------- SEQUENTIAL RECOM --------------------------------')
    sequential_recommendations.main_sequential()