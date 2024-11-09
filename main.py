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
    data = data[data['PlayerId'].isin(id_counts[id_counts >= minimum_match_amount].index)]
    return data

def main():
    data = get_data()
    data = filter_data(data)


if __name__ == '__main__':
    main()