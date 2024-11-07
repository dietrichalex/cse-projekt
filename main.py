import pandas as pd
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


def main():
    data = get_data()
    data = filter_data(data)

if __name__ == '__main__':
    main()
