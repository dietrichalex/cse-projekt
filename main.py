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
    data = data.drop('Age', axis=1)
    data = data.drop('ScoutingReportId', axis=1)
    data = data.drop('ChangedAt',axis=1)

    #filter players with less than 5 matches
    minimum_match_amount = 5
    id_counts = data['PlayerId'].value_counts()
    #data = data[data['PlayerId'].isin(id_counts[id_counts >= minimum_match_amount].index)]

    return data

def print_word_count(data, min_words, max_words, filename):
    filter_words = data['word_count'] = data['Comment'].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0)
    filtered_data = data[data['word_count'] > min_words]
    filtered_data = filtered_data[filtered_data['word_count'] <= max_words]
    # Group by 'ScoutID' and take up to 3 entries per group
    filtered_data = filtered_data.groupby('ScoutId').head(3)

    # Save to a text file
    with open(filename, 'w', encoding='utf-8') as f:
        for idx, row in filtered_data.iterrows():
            f.write(f"-------------------------------------------------------------------------------------------------------------------------------------------------------\n")
            f.write(f"Line: {idx}, ScoutID: {row['ScoutId']}, PlayerID: {row['PlayerId']}, ExactPosition: {row['ExactPosition']} \nComment: {row['Comment']} \nWord Count: {row['word_count']}\n\n")

def main():
    data = get_data()
    data = filter_data(data)
    print_word_count(data,45, 90 ,"filtered_words.txt")


if __name__ == '__main__':
    main()