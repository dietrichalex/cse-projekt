import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_data():
    data = pd.read_csv('data/2013352_dynamic_events_exp.csv', encoding="utf8", delimiter=';')
    data['player_name'] = data['player_name'].str.replace(r'[\x00-\x1F\x7F-\x9F]', '', regex=True)
    return data

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

def plot_heatmap(data, player_id):
    # first and second half in one plot, because it seems like coordinates get inverted at halftime

    player_data = data[data['player_id'] == player_id]
    # only player_possession
    player_data = player_data[player_data['event_type_id'] == 8]

    x = player_data['x_start']/100
    y = player_data['y_start']/100

    plt.figure(figsize=(10, 7))
    sns.kdeplot(x=x, y=y, fill=True, cmap="YlOrRd", thresh=0.05, levels=100)
    plt.title("Heatmap")
    plt.xlim(-55, 55)
    plt.ylim(-35, 35)
    plt.xlabel("")
    plt.ylabel("")
    #plt.gca().set_facecolor('green')
    plt.grid(False)
    plt.show()


def analyse_passing(data, player_id):
    player_data = data[data['player_id'] == player_id]

    # calculate the avg possession time
    player_posstime_data = player_data[player_data['event_type_id'] == 8]
    nof_poss = len(player_posstime_data)
    sum_duration = player_posstime_data['duration'].sum()
    if nof_poss > 0:
        avg_poss = sum_duration / nof_poss
    else:
        avg_poss = 0

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

    print(f"{player_id} has a Pass Accuracy of {pass_accuracy:.2f}% with {total_passes} Passes")
    # Only successful pass are included in this column
    pass_range = player_passes_data['pass_range'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=pass_range.values, y=pass_range.index, hue=pass_range.index, palette="Blues_d", legend=False)
    plt.xlabel("Number of Passes")
    plt.ylabel("Pass Range")
    plt.title("Pass Range Value Counts")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Only successful pass are included in this column
    pass_direction = player_passes_data['pass_direction'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=pass_direction.values, y=pass_direction.index, hue=pass_direction.index, palette="Blues_d", legend=False)
    plt.xlabel("Number of Passes")
    plt.ylabel("Pass Direction")
    plt.title("Pass Direction Value Counts")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def timeline(data, player_id):
    player_data = data[data['player_id'] == player_id]

    # Sort by time
    player_data = player_data.sort_values(by=["period","time_start"])

    # Plot
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
    # select player to analyse
    curr_player = 19175
    # analyse player
    plot_heatmap(data, curr_player)
    analyse_passing(data, curr_player)
    timeline(data, curr_player)

if __name__ == '__main__':
    main()