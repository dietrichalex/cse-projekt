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
    print("Unique Players = ", data['PlayerId'].nunique())
    count = data['PlayerId'].value_counts()
    count_df = count.reset_index(drop=True)  # Resets index to integers

    # Calculate the cumulative sum of occurrences and convert to percentages
    cumulative_occurrences = count_df.cumsum()
    total_occurrences = cumulative_occurrences.iloc[-1]  # Total occurrences at the end of the cumulative sum
    cumulative_percentages = cumulative_occurrences / total_occurrences * 100  # Convert to percentages

    # Identify indices closest to each 10% increment
    percent_steps = np.arange(10, 101, 10)  # Array of target percentages: 10%, 20%, ..., 100%
    xticks = [np.abs(cumulative_percentages - step).idxmin() for step in percent_steps]  # Indices nearest to each 10%
    xtick_labels = [f"{int(cumulative_percentages.iloc[i])}%" for i in xticks]  # Format as percentage labels

    plt.figure(figsize=(24, 8))

    # Plot the data using integer indices for the x-axis
    plt.plot(count_df.index, count_df.values, marker='o')

    # Set labels and title
    plt.xlabel('Cumulative Percentage of Occurrences')
    plt.ylabel('Occurrences')
    plt.title('Players')

    # Apply custom ticks and labels on the x-axis
    plt.xticks(xticks, labels=xtick_labels)

    # Display grid and show plot
    plt.grid(True)
    plt.show()

def plot_scout_occurrence_percentage(data):
    print("Unique Scouts = ", data['ScoutId'].nunique())
    count = data['ScoutId'].value_counts()
    count_df = count.reset_index(drop=True)  # Resets index to integers

    # Calculate the cumulative sum of occurrences and convert to percentages
    cumulative_occurrences = count_df.cumsum()
    total_occurrences = cumulative_occurrences.iloc[-1]  # Total occurrences at the end of the cumulative sum
    cumulative_percentages = cumulative_occurrences / total_occurrences * 100  # Convert to percentages

    # Identify indices closest to each 10% increment
    percent_steps = np.arange(10, 101, 10)  # Array of target percentages: 10%, 20%, ..., 100%
    xticks = [np.abs(cumulative_percentages - step).idxmin() for step in percent_steps]  # Indices nearest to each 10%
    xtick_labels = [f"{int(cumulative_percentages.iloc[i])}%" for i in xticks]  # Format as percentage labels

    plt.figure(figsize=(24, 8))

    # Plot the data using integer indices for the x-axis
    plt.plot(count_df.index, count_df.values, marker='o')

    # Set labels and title
    plt.xlabel('Cumulative Percentage of Occurrences')
    plt.ylabel('Occurrences')
    plt.title('Scouts')

    # Apply custom ticks and labels on the x-axis
    plt.xticks(xticks, labels=xtick_labels)
    plt.grid(True)
    plt.show()

def main():
    data = get_data()
    plot_player_occurrence_percentage(data)
    plot_scout_occurrence_percentage(data)


if __name__ == '__main__':
    main()
