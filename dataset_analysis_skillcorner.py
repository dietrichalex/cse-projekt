import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    data = pd.read_csv('data/2013352_dynamic_events_exp.csv', encoding="utf8", delimiter=';')
    #data.columns = data.columns.str.replace('Column1.', '', regex=False)
    return data


def main():
    data = get_data()
    print(data)

if __name__ == '__main__':
    main()