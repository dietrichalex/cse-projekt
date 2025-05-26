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


def main():
    data = get_data()
    #print(data)

    plot_nof_actions(data, 20)
    plot_heatmap(data, 24470)

if __name__ == '__main__':
    main()