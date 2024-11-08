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
    data = data.drop('ScoutingReportTemplateId', axis=1)
    data = data.drop('ScoutingReportTemplate', axis=1)
    data = data.drop('EventEndDate', axis=1)
    data = data.drop('FilePartition', axis=1)

    return data

def plot_player_occurrence_percentage(data):
    unique_players = data['PlayerId'].nunique()
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

    plt.figure(figsize=(24, 8), dpi=300)

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

    # Add annotation for unique players count
    plt.text(0.95, 0.95, f'Unique Players: {unique_players}',
             transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=14, color='black')

    plt.show()

def plot_scout_occurrence_percentage(data):
    unique_scouts  = data['ScoutId'].nunique()
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

    plt.figure(figsize=(24, 8), dpi=300)

    # Plot the data using integer indices for the x-axis
    plt.plot(count_df.index, count_df.values, marker='o')

    # Set labels and title
    plt.xlabel('Cumulative Percentage of Occurrences')
    plt.ylabel('Occurrences')
    plt.title('Scouts')

    # Apply custom ticks and labels on the x-axis
    plt.xticks(xticks, labels=xtick_labels)
    plt.grid(True)

    # Add annotation for unique players count
    plt.text(0.95, 0.95, f'Unique Scouts: {unique_scouts}',
             transform=plt.gca().transAxes, ha='right', va='bottom', fontsize=14, color='black')

    plt.show()

def plot_rating_occurrence_bar(data):
    count = data['Rating'].value_counts().sort_index()
    plt.figure(figsize=(10, 8),dpi=300)
    ax = count.plot(kind='bar')

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}',
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=12)

    plt.ylabel('Occurrences')
    plt.title('Ratings vs Occurrences')
    plt.show()

def plot_position_occurrence_bar(data):
    count = data['Position'].value_counts()
    plt.figure(figsize=(10, 8),dpi=300)
    ax = count.plot(kind='bar')

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}',
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=12)

    plt.ylabel('Occurrences')
    plt.title('Position vs Occurrences')
    plt.show()

def plot_exact_position_occurrence_bar(data):
    count = data['ExactPosition'].value_counts()
    plt.figure(figsize=(24, 8),dpi=300)
    ax = count.plot(kind='bar')

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}',
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=12)

    plt.ylabel('Occurrences')
    plt.title('Position vs Occurrences')
    plt.show()

def plot_length_of_rating_text_occurrence_bar(data):
    # Calculate the word count in each string, handling non-string entries
    data['word_count'] = data['Comment'].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)

    # Calculate the average, minimum, and maximum word count
    average_word_count = data['word_count'].mean()
    min_word_count = data['word_count'].min()
    max_word_count = data['word_count'].max()

    # Define the bins based on word counts (e.g., < 3 words, < 6 words, etc.)
    bins = [0, 5, 10, 20, 30, 60, 90, 120, 150, float('inf')]  # Adjust the bins as needed
    labels = ['<5', '<10', '<20', '<30', '<60', '<90', '<120', '<150', '>=150']  # Labels for each bin

    # Create a new column for the bin each string belongs to
    data['word_bin'] = pd.cut(data['word_count'], bins=bins, labels=labels, right=False)

    # Count how many strings fall into each bin
    bin_counts = data['word_bin'].value_counts().sort_index()

    # Plot the result as a bar chart
    plt.figure(figsize=(10, 8), dpi=300)
    ax = bin_counts.plot(kind='bar')

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}',
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=12)

    # Add labels and title
    plt.xlabel('Word Count')
    plt.ylabel('Occurrences')
    plt.title('Occurrences in Different Word Count Ranges (whitespace split)')
    plt.xticks(rotation=45)

    # Add annotations for average, min, and max word count
    plt.text(0.95, 0.95, f'Average: {average_word_count:.2f}',
             transform=ax.transAxes, ha='right', fontsize=12)
    plt.text(0.95, 0.90, f'Min: {min_word_count}',
             transform=ax.transAxes, ha='right', fontsize=12)
    plt.text(0.95, 0.85, f'Max: {max_word_count}',
             transform=ax.transAxes, ha='right', fontsize=12)

    # Show the plot
    plt.show()

def plot_data(data):
    plot_player_occurrence_percentage(data)
    plot_scout_occurrence_percentage(data)
    plot_rating_occurrence_bar(data)
    plot_position_occurrence_bar(data)
    plot_exact_position_occurrence_bar(data)
    plot_length_of_rating_text_occurrence_bar(data)

def main():
    data = get_data()
    data = filter_data(data)
    plot_data(data)


if __name__ == '__main__':
    main()
