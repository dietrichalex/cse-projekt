import pandas as pd
import numpy as np

def get_data():
    data = np.array(pd.read_csv('data/Scouting_Reports_FCA.csv', encoding="utf8", delimiter=';'))
    return data

def filter_data(data):
    data = data[:,:-5] #remove unnecessary columns
    #TODO: do more filtering
    return data


def main():
    data = get_data()
    data = filter_data(data)

if __name__ == '__main__':
    main()
