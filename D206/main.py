from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd

# Assess Data Quality

# Read the CSV file
filename = "churn_raw_data.csv"
df = pd.read_csv(filename, keep_default_na = False, na_values=['NA'])
# keep_default_na = False prevents the string "None" from being read as NaN

# Identify missing data
msno.matrix(df)
plt.show()

# Identify duplicate rows based on specified columns
duplicates = df.duplicated(subset=['CaseOrder', 'Customer_id', 'Interaction'], keep=False)

# Display duplicate rows
duplicate_rows = df[duplicates]
print("Duplicate Rows based on 'CaseOrder', 'Customer_id', and 'Interaction':")
print(duplicate_rows)

# Identify data types and non-null count
print(df.info())

# For string (object) columns, use .value_counts()
string_columns = df.select_dtypes(include=["object"]).columns
string_summary = {col: df[col].value_counts() for col in string_columns}

# Iterate through string_summary dictionary to print each string column"s value counts
for col, counts in string_summary.items():
    print(f"Column: {col}")
    print(counts)
    print()

# For numerical (int64/float64) columns, use .describe()
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
numerical_summary = df[numerical_columns].describe()

# Iterate through numerical_summary dictionary to print each string column"s value counts
for col, counts in numerical_summary.items():
    print(f"Column: {col}")
    print(counts)
    print()

# Investigate anomalies in the zip column
df["Zip_str"] = df["Zip"].apply(lambda x: str(int(x)))

# Count the number of anomalies where the length of the zip code is not 5
anomaly_count = (df["Zip_str"].apply(len) != 5).sum()

# Print the number of anomalies
print("Number of zip code anomalies (missing leading zeros):", anomaly_count)

# Investigate outliers in Outage_sec_perweek column
print(df.Outage_sec_perweek.nsmallest(n=20))





# Clean Data
boolean_columns = ["Churn", "Techie", "Port_modem", "Tablet", "Phone", "Multiple", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]

for col in boolean_columns:
    df[col] = df[col].map({"Yes": True, "No": False})

category_columns = ["City", "State", "County", "Area", "Timezone", "Job",
                    "Education", "Employment", "Marital", "Gender", "Contract", "InternetService", "PaymentMethod"]

for col in category_columns:
    df[col] = df[col].astype("category")

df["Zip"] = df["Zip"].astype(str).str.zfill(5)

integer_columns = ["CaseOrder", "Population", "Children", "Age", "Email", "Contacts"]
df[integer_columns] = df[integer_columns].fillna(0).astype(int)

df["Outage_sec_perweek"] = df["Outage_sec_perweek"].abs()

mean_impute_cols = ["Children", "Age", "Income", "Tenure", "Bandwidth_GB_Year"]

for col in mean_impute_cols:
    df[col].fillna(df[col].mean(), inplace=True)

mode_impute_cols = ["Techie", "Phone", "TechSupport"]

for col in mode_impute_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Write to cleaned CSV
df.to_csv("churn_cleaned.csv", index=False)



# Apply PCA

# Numerical columns
numerical_columns = ["Population", "Children", "Age", "Income", "Tenure", "Outage_sec_perweek", "MonthlyCharge", "Bandwidth_GB_Year", "item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8"]

numerical_data = df[numerical_columns].fillna(0)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Initialize and fit PCA
n_components = 16
pca = PCA(n_components=n_components)
pca.fit(scaled_data)

# Create the loading matrix as a DataFrame
loading_matrix = pd.DataFrame(
    pca.components_,
    columns=numerical_columns,
    index=[f"PC{i + 1}" for i in range(n_components)]
)

# Transpose to have principal components as columns and features as rows
transposed_loading_matrix = loading_matrix.T
print(transposed_loading_matrix)


# Calculate eigenvalues
eigenvalues = pca.explained_variance_

# Plot the eigenvalues
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker="o", linestyle="--")
plt.title("Scree Plot of Eigenvalues")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.axhline(y=1, color="red")
plt.grid(True)
plt.show()

print(eigenvalues)