import pandas as pd

df = pd.read_csv("ACSST5Y2022.S1902-Data.csv")

df.columns = df.iloc[0]
df = df[1:]

# Renaming columns to avoid duplicate names issue
df.columns = ['GEO_ID', 'NAME'] + list(df.columns[2:])

# Formatting zip codes in column 'NAME' to remove "ZCTA5 " at the start
df['Zip'] = df['NAME'].str.replace('ZCTA5 ', '').str.zfill(5)


df.to_csv("income_census_clean.csv")