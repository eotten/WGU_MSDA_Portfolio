import pandas as pd

clean_label = lambda x: x.strip()

# Read the CSV file
df = pd.read_csv('A.csv')

# Define the renaming dictionary
rename_dict = {
    "Label (Grouping)": "label",
    "United States!!Total!!Estimate": "total",
    "United States!!Total!!Margin of Error": "total_me",
    "United States!!Percent!!Estimate": "percent",
    "United States!!Percent!!Margin of Error": "percent_me",
    "United States!!Male!!Estimate": "male",
    "United States!!Male!!Margin of Error": "male_me",
    "United States!!Percent Male!!Estimate": "percent_male",
    "United States!!Percent Male!!Margin of Error": "percent_male_me",
    "United States!!Female!!Estimate": "female",
    "United States!!Female!!Margin of Error": "female_me",
    "United States!!Percent Female!!Estimate": "percent_female",
    "United States!!Percent Female!!Margin of Error": "percent_female_me"
}
df.columns = df.columns.map(rename_dict)
df['label'] = df['label'].map(clean_label)

# Split the dataframe table into subtables
df_age = df.iloc[2:20].copy()
df_age_categories = df.iloc[21:33]
df_summary = df.iloc[34:39]
df_percentages = df.iloc[40:42]

# Export subtables to csvs
df_age.to_csv("cleaned/age.csv", index=False)
df_age_categories.to_csv("cleaned/age_categories.csv", index=False)
df_summary.to_csv("cleaned/summary.csv", index=False)
df_percentages.to_csv("cleaned/percentages.csv", index=False)