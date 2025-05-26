import pandas as pd
import requests
import json
import itertools


data = pd.read_csv('data/Scouting_Reports_FCA.csv', encoding="utf8", delimiter=';')
data.columns = data.columns.str.replace('Column1.', '', regex=False)

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

# Für jede ID die Information_X auffüllen
for unique_id in data['PlayerId'].unique():
    # Subset des DataFrames für die aktuelle ID
    subset = data[data['PlayerId'] == unique_id]

    # Vorhandene Werte von Information_X für die aktuelle ID
    available_info = subset['TeamName'].dropna().unique()

    # Falls Werte vorhanden sind, diese auffüllen
    if len(available_info) > 0:
        info_value = available_info[0]  # Nehme den ersten verfügbaren Wert
        data.loc[(data['PlayerId'] == unique_id) & (data['TeamName'].isna()), 'TeamName'] = info_value

# Gefüllten Datensatz speichern
data.to_csv('data/Scouting_Reports_FCA_processed_data.csv', index=False, encoding='utf-8-sig', sep=';')
print(f"Die fehlenden Informationen wurden aufgefüllt und der aktualisierte Datensatz wurde gespeichert: {'data/Scouting_Reports_FCA_processed_data.csv'}")

# Beispielaufruf der Funktion
# fill_missing_information('input.csv', 'output.csv')


# Für jede ID die Information_X auffüllen
for unique_id in data['PlayerId'].unique():
    # Subset des DataFrames für die aktuelle ID
    subset = data[data['PlayerId'] == unique_id]

    # Vorhandene Werte von Information_X für die aktuelle ID
    available_info = subset['Position'].dropna().unique()

    # Falls Werte vorhanden sind, diese auffüllen
    if len(available_info) > 0:
        info_value = available_info[0]  # Nehme den ersten verfügbaren Wert
        data.loc[(data['PlayerId'] == unique_id) & (data['Position'].isna()), 'Position'] = info_value

# Gefüllten Datensatz speichern
data.to_csv('data/Scouting_Reports_FCA_processed_data.csv', index=False, encoding='utf-8-sig', sep=';')
print(f"Die fehlenden Informationen wurden aufgefüllt und der aktualisierte Datensatz wurde gespeichert: {'data/Scouting_Reports_FCA_processed_data.csv'}")

# Beispielaufruf der Funktion
# fill_missing_information('input.csv', 'output.csv')