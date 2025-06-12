import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def get_player_data(path):
    data = pd.read_csv(path, encoding="utf8", delimiter=';')
    data['player_name'] = data['player_name'].str.replace(r'[\x00-\x1F\x7F-\x9F]', '', regex=True)
    return data

def get_my_data(path):
    data = pd.read_csv(path, encoding="utf8", delimiter=';', decimal=',')
    return data

def get_nof_actions(data):
    return len(data)

def get_nof_games(data):
    return len(data['match_id'].unique())

def get_passing_data(data):
    # Count total and successful passes
    passes_data = data[data['end_type_id'] == 1]
    nof_passes = len(passes_data)
    successful_passes = len(passes_data[passes_data['pass_outcome'] == 'successful'])
    # Calculate accuracy
    if nof_passes > 0:
        pass_accuracy = (successful_passes / nof_passes) * 100
    else:
        pass_accuracy = 0.0

    # Only successful pass are included in this column
    #pass_direction_value = passes_data['pass_angle'].mean()
    passes_forward = len(passes_data[passes_data['pass_direction'] == 'forward'])/nof_passes*100
    passes_backward = len(passes_data[passes_data['pass_direction'] == 'backward'])/nof_passes*100
    passes_left = len(passes_data[passes_data['pass_direction'] == 'sideway_left'])/nof_passes*100
    passes_right = len(passes_data[passes_data['pass_direction'] == 'sideway_right'])/nof_passes*100
    pass_range_value = passes_data['pass_distance'].mean() / 100

    # Calculate dangerous accuracy
    dangerous_passes_data = passes_data[passes_data['player_targeted_dangerous'] == 'WAHR']
    nof_dangerous_passes = len(dangerous_passes_data)
    successful_dangerous_passes = len(dangerous_passes_data[dangerous_passes_data['pass_outcome'] == 'successful'])
    if nof_dangerous_passes > 0:
        dangerous_pass_accuracy = (successful_dangerous_passes / nof_dangerous_passes) * 100
    else:
        dangerous_pass_accuracy = 0.0

    # Calculate difficult accuracy
    difficult_passes_data = passes_data[passes_data['player_targeted_difficult_pass_target'] == 'WAHR']
    nof_difficult_passes = len(difficult_passes_data)
    successful_difficult_passes = len(difficult_passes_data[difficult_passes_data['pass_outcome'] == 'successful'])
    if nof_difficult_passes > 0:
        difficult_pass_accuracy = (successful_difficult_passes / nof_difficult_passes) * 100
    else:
        difficult_pass_accuracy = 0.0

    # successful linebreak passes
    nof_succesful_first_linebreakpasses = len(passes_data[(passes_data['first_line_break'] == 'WAHR')])
    nof_succesful_secondlast_linebreakpasses = len(passes_data[(passes_data['second_last_line_break'] == 'WAHR')])
    nof_succesful_last_linebreakpasses = len(passes_data[(passes_data['last_line_break'] == 'WAHR')])

    return nof_passes, pass_accuracy, passes_forward, passes_backward, passes_right, passes_left, pass_range_value, nof_dangerous_passes, dangerous_pass_accuracy, nof_difficult_passes, difficult_pass_accuracy, nof_succesful_first_linebreakpasses, nof_succesful_secondlast_linebreakpasses, nof_succesful_last_linebreakpasses

def get_possession_data(data):
    # calculate the avg possession time
    posstime_data = data[data['event_type_id'] == 8]
    nof_poss = len(posstime_data)
    sum_duration = posstime_data['duration'].sum()
    if nof_poss > 0:
        avg_poss = sum_duration / nof_poss
    else:
        avg_poss = 0

    nof_carrys = len(posstime_data[posstime_data['carry'] == 'WAHR'])

    return avg_poss, nof_carrys

def get_nof_chances_created(data):
    # Without filtering after possession, for example off_ball_runs would also count
    possession_data = data[data['event_type_id'] == 8]
    return len(possession_data[possession_data['lead_to_shot'] == 'WAHR'])

def get_nof_goal_created(data):
    # Without filtering after possession, for example off_ball_runs would also count
    possession_data = data[data['event_type_id'] == 8]
    return len(possession_data[possession_data['lead_to_goal'] == 'WAHR'])

def create_mydata(data):
    players = data['player_id'].unique()
    counter = 1
    max_counter = len(players)
    mydata = []
    for player in players:
        print(f"Analysing {player} ({counter}/{max_counter})")
        player_data = data[data['player_id'] == player]
        player_name = player_data['player_name'].iloc[0]
        player_position = player_data['player_position'].iloc[0]
        nof_games = get_nof_games(player_data)
        nof_actions = get_nof_actions(player_data)
        nof_passes, pass_accuracy, passes_forward, passes_backward, passes_right, passes_left, pass_range_value, nof_dangerous_passes, dangerous_pass_accuracy, nof_difficult_passes, difficult_pass_accuracy, nof_succesful_first_linebreakpasses, nof_succesful_secondlast_linebreakpasses, nof_succesful_last_linebreakpasses = get_passing_data(player_data)
        avg_poss, nof_carrys = get_possession_data(player_data)
        nof_chances_created = get_nof_chances_created(player_data)
        nof_goal_created = get_nof_goal_created(player_data)

        counter += 1
        mydata.append({"player_name": player_name,
                       "player_id": player,
                       "player_position": player_position,
                       "number_of_games": nof_games,
                       "avg_number_of_actions_per_game": nof_actions/nof_games,
                       "avg_number_of_passes_per_game": nof_passes/nof_games,
                       "pass_accuracy_%": pass_accuracy,
                       "avg_pass_range_m": pass_range_value,
                       "passes_forward_%": passes_forward,
                       "passes_backward_%": passes_backward,
                       "passes_right_%": passes_right,
                       "passes_left_%": passes_left,
                       "avg_poss_duration_s": avg_poss,
                       "avg_number_of_dangerous_passes_per_game": nof_dangerous_passes/nof_games,
                       "dangerous_pass_accuracy_%": dangerous_pass_accuracy,
                       "avg_number_of_difficult_passes_per_game": nof_difficult_passes/nof_games,
                       "difficult_pass_accuracy_%": difficult_pass_accuracy,
                       "number_of_possession_lead_to_shot_per_game": nof_chances_created/nof_games,
                       "number_of_possession_lead_to_goal_per_game": nof_goal_created/nof_games,
                       "number_of_successful_first_linebreakpasses_per_game": nof_succesful_first_linebreakpasses/nof_games,
                       "number_of_successful_secondlast_linebreakpasses_per_game": nof_succesful_secondlast_linebreakpasses/nof_games,
                       "number_of_successful_last_linebreakpasses_per_game": nof_succesful_last_linebreakpasses/nof_games,
                       "number_of_carrys_per_game": nof_carrys/nof_games,
                       })

    df = pd.DataFrame(mydata)
    # Convert float columns to string with comma decimal separator
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].map(lambda x: str(x).replace('.', ','))
    df.to_csv("data/mydata.csv",
              sep=";",
              index=False,
              float_format="%.3f",
              encoding="utf-8"
              )

def calc_similarity_score(data):
    filtered_data = data.iloc[:, 3:]
    scaled_filtered_data = filtered_data/filtered_data.max()
    n = scaled_filtered_data.shape[0]
    out = np.zeros((n, n))
    for i in range(len(scaled_filtered_data)):
        row_i_array = scaled_filtered_data.iloc[i].to_numpy()
        for j in range(i,len(scaled_filtered_data)):
            row_j_array = scaled_filtered_data.iloc[j].to_numpy()
            print(f"Calculating Similarity-Score of {i} to {j}")
            sim_score = cosine_similarity(row_i_array.reshape(1, -1), row_j_array.reshape(1, -1))[0][0]
            out[i, j] = sim_score
            out[j, i] = sim_score

    df = pd.DataFrame(out)
    # Convert float columns to string with comma decimal separator
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].map(lambda x: str(x).replace('.', ','))
    df.to_csv("data/similarity_score_matrix.csv",
              index=False,
              header=False,
              sep=";",
              float_format="%.3f",
              encoding="utf-8",
              )

def main():
    # get data
    data = get_player_data('data/2013352_dynamic_events_exp.csv')
    create_mydata(data)
    mydata = get_my_data('data/mydata.csv')
    calc_similarity_score(mydata)

if __name__ == '__main__':
    main()