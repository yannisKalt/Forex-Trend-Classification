import os
import pandas as pd

# get currency pair csv names.
csv_names = [nm for nm in os.listdir() if nm.endswith('.csv')]


def gmt_to_timestamp(gmt_str):
    date, time = gmt_str.split(' ')
    day, month, year = [int(x) for x in date.split('.')]
    hour = int(time.split(':')[0])
    return pd.Timestamp(year = year, month = month,
                        day = day, hour = hour)

# load pairs and change Time column to pd.Timestamp
for pair_csv_name in csv_names:
        data = pd.read_csv(pair_csv_name, index_col = 0, sep = ',')
        gmt_time = data.index 
        pair_code = pair_csv_name.split('.')[0]
        
        altered_time_index = pd.Series(map(gmt_to_timestamp, gmt_time))
        data.index = altered_time_index
        data.index.name = 'Timestamp'
        data.columns = [pair_code + '_' + s for s in data.columns]
         
        data.to_csv(pair_csv_name)

