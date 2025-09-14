import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import glob
from pathlib import Path


class Timer:
    def __init__(self):
        self.times = {}

    def start(self, name):
        self.times[name] = time.perf_counter()

    def stop(self, name):
        if name not in self.times:
            print(f"Timer '{name}' was never started")
            return None

        elapsed = time.perf_counter() - self.times[name]
        print(f"{name}: {elapsed:.4f} seconds")
        return elapsed

    def get_summary(self):
        print("\n--- Timing Summary ---")
        for name, elapsed in self.times.items():
            print(f"{name}: {elapsed:.4f} seconds")


def get_player_data_from_csv_folder(folder_path):
    """
    Iterate over all CSV files in the specified folder and combine the data
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")

    # Get all CSV files in the folder
    csv_files = list(folder_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {file.name}")

    all_data = []

    for file_path in csv_files:
        print(f"\nProcessing: {file_path.name}")
        try:
            # Try to read the CSV file with same settings as original function
            data = pd.read_csv(
                file_path,
                encoding="utf8",
                delimiter=',',
                dtype=str  # pandas nullable integer type
            )

            # Clean player_name from control characters if the column exists
            if 'player_name' in data.columns:
                data['player_name'] = data['player_name'].str.replace(
                    r'[\x00-\x1F\x7F-\x9F]', '', regex=True
                )

            print(f"  Loaded {len(data)} rows from {file_path.name}")
            all_data.append(data)

        except Exception as e:
            print(f"  Error reading {file_path.name}: {str(e)}")
            continue

    if not all_data:
        raise ValueError("No data could be loaded from any CSV files")

    # Combine all dataframes
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined data: {len(combined_data)} total rows from {len(all_data)} files")

    return combined_data


def get_player_data_from_csv(path):
    """
    Original function for CSV files (kept for backward compatibility)
    """
    # Force player_id as int, others will be auto-detected
    data = pd.read_csv(
        path,
        encoding="utf8",
        delimiter=';',
        dtype={'player_id': 'Int64'}  # pandas nullable integer type
    )
    # Clean player_name from control characters
    data['player_name'] = data['player_name'].str.replace(
        r'[\x00-\x1F\x7F-\x9F]', '', regex=True
    )
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
    passes_data = data[data['end_type_id'] == '1.0']
    nof_passes = len(passes_data)
    successful_passes = len(passes_data[passes_data['pass_outcome'] == 'successful'])
    # Calculate accuracy
    if nof_passes > 0:
        pass_accuracy = (successful_passes / nof_passes) * 100
        passes_forward = len(passes_data[passes_data['pass_direction'] == 'forward']) / nof_passes * 100
        passes_backward = len(passes_data[passes_data['pass_direction'] == 'backward']) / nof_passes * 100
        passes_left = len(passes_data[passes_data['pass_direction'] == 'sideway_left']) / nof_passes * 100
        passes_right = len(passes_data[passes_data['pass_direction'] == 'sideway_right']) / nof_passes * 100
    else:
        pass_accuracy = 0.0
        passes_forward = 0.0
        passes_backward = 0.0
        passes_left = 0.0
        passes_right = 0.0

    # Only successful pass are included in this column
    # pass_direction_value = passes_data['pass_angle'].mean()
    pass_range_value = passes_data['pass_distance'].astype(float).dropna().astype(int).mean()

    # Calculate dangerous accuracy
    dangerous_passes_data = passes_data[passes_data['player_targeted_dangerous'] == 'True']
    nof_dangerous_passes = len(dangerous_passes_data)
    successful_dangerous_passes = len(dangerous_passes_data[dangerous_passes_data['pass_outcome'] == 'successful'])
    if nof_dangerous_passes > 0:
        dangerous_pass_accuracy = (successful_dangerous_passes / nof_dangerous_passes) * 100
    else:
        dangerous_pass_accuracy = 0.0

    # Calculate difficult accuracy
    difficult_passes_data = passes_data[passes_data['player_targeted_difficult_pass_target'] == 'True']
    nof_difficult_passes = len(difficult_passes_data)
    successful_difficult_passes = len(difficult_passes_data[difficult_passes_data['pass_outcome'] == 'successful'])
    if nof_difficult_passes > 0:
        difficult_pass_accuracy = (successful_difficult_passes / nof_difficult_passes) * 100
    else:
        difficult_pass_accuracy = 0.0

    # successful linebreak passes
    nof_succesful_first_linebreakpasses = len(passes_data[(passes_data['first_line_break'] == 'True')])
    nof_succesful_secondlast_linebreakpasses = len(passes_data[(passes_data['second_last_line_break'] == 'True')])
    nof_succesful_last_linebreakpasses = len(passes_data[(passes_data['last_line_break'] == 'True')])

    return nof_passes, pass_accuracy, passes_forward, passes_backward, passes_right, passes_left, pass_range_value, nof_dangerous_passes, dangerous_pass_accuracy, nof_difficult_passes, difficult_pass_accuracy, nof_succesful_first_linebreakpasses, nof_succesful_secondlast_linebreakpasses, nof_succesful_last_linebreakpasses


def get_possession_data(data):
    # calculate the avg possession time
    posstime_data = data[data['event_type_id'] == '8']
    nof_poss = len(posstime_data)
    sum_duration = posstime_data['duration'].astype(float).dropna().sum()
    if nof_poss > 0:
        avg_poss = sum_duration / nof_poss
    else:
        avg_poss = 0

    nof_carrys = len(posstime_data[posstime_data['carry'] == 'True'])

    return avg_poss, nof_carrys


def get_nof_chances_created(data):
    # Without filtering after possession, for example off_ball_runs would also count
    possession_data = data[data['event_type_id'] == '8']
    return len(possession_data[possession_data['lead_to_shot'] == 'True'])


def get_nof_goal_created(data):
    # Without filtering after possession, for example off_ball_runs would also count
    possession_data = data[data['event_type_id'] == '8']
    return len(possession_data[possession_data['lead_to_goal'] == 'True'])


def get_nof_being_pass_option(data):
    return len(data[data['event_type_id'] == '7'])


def get_nof_off_ball_runs(data):
    return len(data[data['event_type_id'] == '1'])


def get_nof_shots(data):
    return len(data[data['end_type_id'] == '2.0'])


def get_nof_goals(data):
    return len(data[(data['end_type_id'] == '2.0') & (data['lead_to_goal'] == 'True')])


def get_max_avg_speed(data):
    return data["speed_avg"].astype(float).dropna().astype(int).max()


def get_nof_interceptions(data):
    return len(data[data['start_type_id'] == '2.0']), len(data[data['start_type_id'] == '6.0']), len(
        data[data['start_type_id'] == '10.0']), len(data[data['start_type_id'] == '12.0'])


def get_nof_clearences(data):
    return len(data[data['end_type_id'] == '3.0'])


def get_nof_recoverys(data):
    return len(data[data['start_type_id'] == '4.0'])


def create_mydata(data):
    players = data['player_id'].astype(float).astype(int).unique()
    counter = 1
    max_counter = len(players)
    mydata = []
    for player in players:
        print(f"Analysing {player} ({counter}/{max_counter})")
        player_data = data[data['player_id'].astype(float).astype(int) == player]
        player_name = player_data['player_name'].iloc[0]
        player_position = player_data['player_position'].iloc[0]
        nof_games = get_nof_games(player_data)
        nof_actions = get_nof_actions(player_data)
        nof_passes, pass_accuracy, passes_forward, passes_backward, passes_right, passes_left, pass_range_value, nof_dangerous_passes, dangerous_pass_accuracy, nof_difficult_passes, difficult_pass_accuracy, nof_succesful_first_linebreakpasses, nof_succesful_secondlast_linebreakpasses, nof_succesful_last_linebreakpasses = get_passing_data(
            player_data)
        avg_poss, nof_carrys = get_possession_data(player_data)
        nof_chances_created = get_nof_chances_created(player_data)
        nof_goal_created = get_nof_goal_created(player_data)
        nof_being_pass_opt = get_nof_being_pass_option(player_data)
        nof_off_ball_runs = get_nof_off_ball_runs(player_data)
        nof_shots = get_nof_shots(player_data)
        nof_goals = get_nof_goals(player_data)
        max_avg_speed = get_max_avg_speed(player_data)
        nof_pass_interceptions, nof_freekick_interceptions, nof_goalkick_interceptions, nof_corner_interceptions = get_nof_interceptions(
            player_data)
        nof_clearences = get_nof_clearences(player_data)
        nof_recoverys = get_nof_recoverys(player_data)

        counter += 1
        mydata.append({"player_name": player_name,
                       "player_id": player,
                       "player_position": player_position,
                       "number_of_games": nof_games,
                       "avg_number_of_actions_per_game": nof_actions / nof_games,
                       "avg_number_of_passes_per_game": nof_passes / nof_games,
                       "pass_accuracy_%": pass_accuracy,
                       "avg_pass_range_m": pass_range_value,
                       "passes_forward_%": passes_forward,
                       "passes_backward_%": passes_backward,
                       "passes_right_%": passes_right,
                       "passes_left_%": passes_left,
                       "avg_poss_duration_s": avg_poss,
                       "avg_number_of_dangerous_passes_per_game": nof_dangerous_passes / nof_games,
                       "dangerous_pass_accuracy_%": dangerous_pass_accuracy,
                       "avg_number_of_difficult_passes_per_game": nof_difficult_passes / nof_games,
                       "difficult_pass_accuracy_%": difficult_pass_accuracy,
                       "number_of_possession_lead_to_shot_per_game": nof_chances_created / nof_games,
                       "number_of_possession_lead_to_goal_per_game": nof_goal_created / nof_games,
                       "number_of_successful_first_linebreakpasses_per_game": nof_succesful_first_linebreakpasses / nof_games,
                       "number_of_successful_secondlast_linebreakpasses_per_game": nof_succesful_secondlast_linebreakpasses / nof_games,
                       "number_of_successful_last_linebreakpasses_per_game": nof_succesful_last_linebreakpasses / nof_games,
                       "number_of_carrys_per_game": nof_carrys / nof_games,
                       "number_of_being_passing_option_per_game": nof_being_pass_opt,
                       "number_of_off_ball_runs_per_game": nof_off_ball_runs,
                       "number_of_shots_per_game": nof_shots / nof_games,
                       "number_of_goals_per_game": nof_goals / nof_games,
                       "maximum_average_speed_kmh": max_avg_speed,
                       "number_of_pass_interceptions_per_game": nof_pass_interceptions / nof_games,
                       "number_of_freekick_interceptions_per_game": nof_freekick_interceptions / nof_games,
                       "number_of_goalkick_interceptions_per_game": nof_goalkick_interceptions / nof_games,
                       "number_of_corner_interceptions_per_game": nof_corner_interceptions / nof_games,
                       "number_of_clearences_per_game": nof_clearences / nof_games,
                       "number_of_recoverys_per_game": nof_recoverys / nof_games,
                       })

    df = pd.DataFrame(mydata)
    filtered_data = df.iloc[:, 3:]
    scaled_filtered_data = filtered_data / filtered_data.max()
    # calculate the categories
    passing_parameters = ["avg_number_of_passes_per_game",
                          "pass_accuracy_%",
                          "avg_pass_range_m",
                          "passes_forward_%",
                          "passes_backward_%",
                          "passes_right_%",
                          "passes_left_%",
                          "avg_number_of_dangerous_passes_per_game",
                          "dangerous_pass_accuracy_%",
                          "avg_number_of_difficult_passes_per_game",
                          "difficult_pass_accuracy_%",
                          "number_of_successful_first_linebreakpasses_per_game",
                          "number_of_successful_secondlast_linebreakpasses_per_game",
                          "number_of_successful_last_linebreakpasses_per_game", ]
    posession_parameters = ["avg_number_of_actions_per_game",
                            "avg_poss_duration_s",
                            "number_of_possession_lead_to_shot_per_game",
                            "number_of_possession_lead_to_goal_per_game",
                            "number_of_shots_per_game",
                            "number_of_goals_per_game", ]
    off_ball_parameters = ["number_of_being_passing_option_per_game",
                           "number_of_recoverys_per_game", ]
    defensive_parameters = ["number_of_pass_interceptions_per_game",
                            "number_of_freekick_interceptions_per_game",
                            "number_of_goalkick_interceptions_per_game",
                            "number_of_corner_interceptions_per_game",
                            "number_of_clearences_per_game", ]
    physical_parameters = ["number_of_carrys_per_game",
                           "number_of_off_ball_runs_per_game",
                           "maximum_average_speed_kmh"]
    df["passing_parameters"] = scaled_filtered_data[passing_parameters].mean(axis=1)
    df["posession_parameters"] = scaled_filtered_data[posession_parameters].mean(axis=1)
    df["off_ball_parameters"] = scaled_filtered_data[off_ball_parameters].mean(axis=1)
    df["defensive_parameters"] = scaled_filtered_data[defensive_parameters].mean(axis=1)
    df["physical_parameters"] = scaled_filtered_data[physical_parameters].mean(axis=1)
    df = df.fillna(0)

    # Round all numeric columns to 3 decimal places BEFORE converting to string
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(3)

    # Convert float columns to string with comma decimal separator
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].map(lambda x: f"{x:.3f}".replace('.', ','))

    df.to_csv("data/mydata.csv",
              sep=";",
              index=False,
              encoding="utf-8"
              )


def calc_similarity_score(data, weights, flg_default):
    scaled_data = data / data.max()
    n = scaled_data.shape[0]
    out = np.zeros((n, n))
    for i in range(len(scaled_data)):
        row_i_array = scaled_data.iloc[i].to_numpy()
        for j in range(i, len(scaled_data)):
            sim_score = 0
            row_j_array = scaled_data.iloc[j].to_numpy()
            print(f"Calculating Similarity-Score of {i} to {j}")
            for k in range(len(row_i_array)):
                sim_score += weights[k] * abs(row_i_array[k] - row_j_array[k])
            sim_score = sim_score / np.sum(weights)
            out[i, j] = 1 - sim_score
            out[j, i] = 1 - sim_score

    df = pd.DataFrame(out)

    # Round all values to 3 decimal places BEFORE converting to string
    df = df.round(3)

    # Convert float columns to string with comma decimal separator
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].map(lambda x: f"{x:.3f}".replace('.', ','))

    if flg_default:
        df.to_csv("data/similarity_score_matrix.csv",
                  index=False,
                  header=False,
                  sep=";",
                  encoding="utf-8",
                  )
    else:
        df.to_csv("data/similarity_score_matrix_weights.csv",
                  index=False,
                  header=False,
                  sep=";",
                  encoding="utf-8",
                  )


def main():
    timer = Timer()

    # Create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Define the subfolder path for CSV files
    csv_folder = "data/game_files"  # Change this path as needed

    # Check if we should use CSV files from folder or single CSV
    if os.path.exists(csv_folder):
        print(f"CSV folder found: {csv_folder}")
        timer.start("Data Loading from CSV files")
        data = get_player_data_from_csv_folder(csv_folder)
        timer.stop("Data Loading from CSV files")
    elif os.path.exists('data/merged.csv'):
        print("Using existing CSV file: data/merged.csv")
        timer.start("Data Loading from CSV")
        data = get_player_data_from_csv('data/merged.csv')
        timer.stop("Data Loading from CSV")
    else:
        raise FileNotFoundError("Neither CSV folder 'data/game_files' nor CSV file 'data/merged.csv' found")

    # create myData
    timer.start("Create myData")
    create_mydata(data)
    timer.stop("Create myData")

    # create SimScoreMatrix
    timer.start("Calculate Similarity Score")
    mydata = get_my_data('data/mydata.csv')
    filtered_data = mydata.iloc[:, 3:]
    filtered_data = filtered_data.iloc[:, :-5]
    weights = np.ones(filtered_data.shape[1], )
    calc_similarity_score(filtered_data, weights, True)
    timer.stop("Calculate Similarity Score")

    timer.get_summary()


if __name__ == '__main__':
    main()