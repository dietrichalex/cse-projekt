import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def get_data():
    data = pd.read_csv('data/2013352_dynamic_events_exp.csv', encoding="utf8", delimiter=';')
    data['player_name'] = data['player_name'].str.replace(r'[\x00-\x1F\x7F-\x9F]', '', regex=True)
    return data

def get_sim_score(data, player_id_1, player_id_2):
    # Number of actions
    player_1_nof_actions = get_nof_actions(data, player_id_1)
    player_2_nof_actions = get_nof_actions(data, player_id_2)
    # Plot Heatmaps
    plot_heatmap(data, player_id_1, False)
    plot_heatmap(data, player_id_2, False)
    # Analyse passing
    player_1_pass_range, player_1_pass_direction, player_1_avg_poss, player_1_pass_accuracy = analyse_passing(data, player_id_1, False)
    player_2_pass_range, player_2_pass_direction, player_2_avg_poss, player_2_pass_accuracy = analyse_passing(data, player_id_2, False)
    # Timeline
    timeline(data, player_id_1, False)
    timeline(data, player_id_2, False)

    # Calculate similarity score
    player_1 = np.array([player_1_nof_actions, player_1_pass_range, player_1_pass_direction, player_1_avg_poss, player_1_pass_accuracy])
    player_2 = np.array([player_2_nof_actions, player_2_pass_range, player_2_pass_direction, player_2_avg_poss, player_2_pass_accuracy])
    sim_score = cosine_similarity(player_1.reshape(1, -1), player_2.reshape(1, -1))[0][0]
    print(f"Similarity score (0–1): {sim_score:.3f}")
    return sim_score

def get_nof_actions(data, player_id):
    player_data = data[data['player_id'] == player_id]
    nof_actions = len(player_data)
    return nof_actions

def plot_nof_actions(data, nof_players):
    # only player_possession
    poss_data = data[data['event_type_id'] == 8]
    act_player = poss_data['player_name'].value_counts()

    top_player = act_player.head(nof_players)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_player.values, y=top_player.index, hue=top_player.index, palette="Blues_d", legend=False)
    plt.xlabel("Number of Actions")
    plt.ylabel("Player")
    plt.title("Top 10 Player Possessions")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_heatmap(data, player_id, flg_plot):
    # first and second half in one plot, because it seems like coordinates get inverted at halftime

    player_data = data[data['player_id'] == player_id]
    # only player_possession
    player_data = player_data[player_data['event_type_id'] == 8]

    x = player_data['x_start']/100
    y = player_data['y_start']/100

    if flg_plot:
        plt.figure(figsize=(10, 7))
        sns.kdeplot(x=x, y=y, fill=True, cmap="YlOrRd", thresh=0.05, levels=100)
        plt.title(f"Heatmap for {player_id}")
        plt.xlim(-55, 55)
        plt.ylim(-35, 35)
        plt.xlabel("")
        plt.ylabel("")
        #plt.gca().set_facecolor('green')
        plt.grid(False)
        plt.show()


def analyse_passing(data, player_id, flg_plot):
    player_data = data[data['player_id'] == player_id]

    # calculate the avg possession time
    player_posstime_data = player_data[player_data['event_type_id'] == 8]
    nof_poss = len(player_posstime_data)
    sum_duration = player_posstime_data['duration'].sum()
    if nof_poss > 0:
        avg_poss = sum_duration / nof_poss
    else:
        avg_poss = 0

    if flg_plot:
        print(f"{player_id} has a average Possession time of {avg_poss:.2f}s with {nof_poss} Possessions")

    # Count total and successful passes
    player_passes_data = player_data[player_data['end_type_id'] == 1]
    total_passes = len(player_passes_data)
    successful_passes = len(player_passes_data[player_passes_data['pass_outcome'] == 'successful'])

    # Calculate accuracy
    if total_passes > 0:
        pass_accuracy = (successful_passes / total_passes) * 100
    else:
        pass_accuracy = 0.0

    if flg_plot:
        print(f"{player_id} has a Pass Accuracy of {pass_accuracy:.2f}% with {total_passes} Passes")
    # Only successful pass are included in this column
    pass_range = player_passes_data['pass_range'].value_counts()
    pass_range_value = player_passes_data['pass_distance'].mean()/100

    if flg_plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=pass_range.values, y=pass_range.index, hue=pass_range.index, palette="Blues_d", legend=False)
        plt.xlabel("Number of Passes")
        plt.ylabel("Pass Range")
        plt.title(f"Pass Range Value Counts for {player_id}")
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # Only successful pass are included in this column
    pass_direction = player_passes_data['pass_direction'].value_counts()
    pass_direction_value = player_passes_data['pass_angle'].mean()

    if flg_plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=pass_direction.values, y=pass_direction.index, hue=pass_direction.index, palette="Blues_d", legend=False)
        plt.xlabel("Number of Passes")
        plt.ylabel("Pass Direction")
        plt.title(f"Pass Direction Value Counts for {player_id}")
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return pass_range_value, pass_direction_value, avg_poss, pass_accuracy



def timeline(data, player_id, flg_plot):
    player_data = data[data['player_id'] == player_id]

    # Sort by time
    player_data = player_data.sort_values(by=["period","time_start"])

    # Plot
    if flg_plot:
        plt.figure(figsize=(12, 4))

        plt.scatter(player_data["time_start"], player_data["event_type"], c='blue', alpha=0.6)

        plt.title(f"Event Timeline for {player_id}")
        plt.xlabel("Time")
        plt.ylabel("Event Type")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    # get data
    data = get_data()
    #print(data)
    plot_nof_actions(data, 20)
    # select players to analyse
    player_1 = 12484
    player_2 = 4047
    # analyse players
    sim_score = get_sim_score(data, player_1, player_2)

if __name__ == '__main__':
    main()