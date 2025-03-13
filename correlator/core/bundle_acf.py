
import os
import pandas as pd

BASE = r'D:\Analysis\FCS\20241120'

for exp in os.listdir(BASE):
    folder = f"{BASE}\\{exp}"

    # == Bundle experiment folder
    if not os.path.isdir(folder): 
        print('skipping', exp)
        continue

    csv_dir = f"{folder}\\csv"
    df = pd.DataFrame()
    for i, name in enumerate(os.listdir(csv_dir)):
        # == Read individual acf data
        data = pd.read_csv(f"{csv_dir}\\{name}", index_col=0)
        if i == 0:
            df['tau'] = data['tau']
        df[name] = data['acf']

    df.to_csv(f"{folder}.csv", index=False)

print(f"\nFinished successfully")