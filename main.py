import csv
import pandas as pd
import numpy as np

def get_data():
   with open('data/Scouting_Reports_FCA.csv', encoding="utf8") as csvfile:
       csv_reader = csv.reader(csvfile, delimiter=';')
       for row in csv_reader:
           print(row)


def main():
    get_data()

if __name__ == '__main__':
    main()
