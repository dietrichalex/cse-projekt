import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    data = pd.read_csv('data/Scouting_Reports_FCA.csv', encoding="utf8", delimiter=';')
    data.columns = data.columns.str.replace('Column1.', '', regex=False)
    return data

def filter_data(data):
    # remove unnecessary columns
    data = data.iloc[:,:-5]

    #filter players with less than 10 matches
    minimum_match_amount = 10
    id_counts = data['PlayerId'].value_counts()
    data = data[data['PlayerId'].isin(id_counts[id_counts >= minimum_match_amount].index)]
    return data

def plot_player_occurrence_percentage(data):
    count = data['PlayerId'].value_counts()
    count.plot(kind='line', marker='o')
    plt.xlabel('Amount of Data in Percentage')
    plt.ylabel('Occurrences')
    plt.title('Occurrences per Percentage')
    xticks = range(0, len(count), max(1, len(count) // 10))
    xtick_labels = [f"{(i / len(count)) * 100:.0f}%" for i in xticks]
    plt.xticks(xticks, labels=xtick_labels)
    plt.grid(True)
    plt.show()

def main():
    data = get_data()
    plot_player_occurrence_percentage(data)
    data = filter_data(data)


if __name__ == '__main__':
    main()
