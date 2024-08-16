from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def backward_elimination(X, Y, p_value_threshold = 0.05):
    def calculate_vif(X):
        """Helper function to calculate VIFs for features in a given dataset."""
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(X.shape[1])
        ]
        return vif_data

    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    round_count = 0
    while True:
        # Fit the model
        print(f"\nRound {round_count} of backward elimination:")
        model = sm.OLS(Y, X).fit()
        print(model.summary())

        # Check for high p-values
        high_p_value = model.pvalues[model.pvalues > p_value_threshold]

        """
        # Calculate VIFs
        vif_data = calculate_vif(X)
        high_vif = vif_data[vif_data["VIF"] > vif_threshold]

        # Check conditions to remove: high VIF and, if applicable, high p-value
        if high_vif.empty and high_p_value.empty:
            print("p-values:", model.pvalues)
            break

        # Prefer to remove high VIF variables first
        if not high_vif.empty:
            # Find the variable with the highest VIF
            feature_to_remove = high_vif.sort_values("VIF", ascending=False).iloc[0][
                "feature"
            ]
        elif not high_p_value.empty:
            # Or remove the least significant variable (highest p-value)
            feature_to_remove = high_p_value.idxmax()
        """

        if (model.pvalues < p_value_threshold).all():
            print("p-values:", model.pvalues)
            break
        else:
            # Or remove the least significant variable (highest p-value)
            feature_to_remove = high_p_value.idxmax()

        # Drop the feature with the highest VIF or p-value
        X = X.drop(columns=[feature_to_remove])
        print("Removing feature:", feature_to_remove)
        round_count += 1

    return X, model


# Read the CSV file
filename = "churn_clean.csv"
df = pd.read_csv(filename, keep_default_na=False, na_values=["NA"])

input("Press enter to begin Part C2")
# C2.  Dependent and All Independent Variables Summary Statistics

# Temporarily include 'CaseOrder' and 'Zip' as categorical variables
categorical_cols = df.select_dtypes(include=["object", "bool"]).copy()
categorical_cols['CaseOrder'] = df['CaseOrder'].astype('category')
categorical_cols['Zip'] = df['Zip'].astype('category')

# Update numerical columns to exclude 'CaseOrder' and 'Zip' if they were included
numerical_cols = df.select_dtypes(include=["int64", "float64"]).drop(columns=['CaseOrder', 'Zip'], errors='ignore')


# Generate summaries
categorical_summary = {col: df[col].value_counts() for col in categorical_cols}
numerical_summary = {col: df[col].describe() for col in numerical_cols}

# Display the results
print("Categorical Data Summary:")
for key, value in categorical_summary.items():
    print(f"\nColumn: {key}\n{value}")

print("\nNumerical Data Summary:")
for key, value in numerical_summary.items():
    print(f"\nColumn: {key}\n{value}")

input("Press enter to begin Part C3")
# C3.  Univariate and Bivariate Visualizations
plt.figure(figsize=(12, 6))
plt.hist(df["Tenure"], bins=30, color="green", edgecolor="black")
plt.title("Distribution of Customer Tenure")
plt.xlabel("Tenure (months)")
plt.ylabel("Number of Customers")
plt.grid(True)
plt.show()

numerical_column_titles = {
    # "CaseOrder":             "Case Order",
    "Lat": "Latitude",
    "Lng": "Longitude",
    "Population": "Area Population",
    "Children": "Number of Children",
    "Age": "Customer Age",
    "Income": "Customer Income",
    "Outage_sec_perweek": "Outage Seconds Per Week",
    "Email": "Number of Emails Sent",
    "Contacts": "Number of Support Contacts",
    "Yearly_equip_failure": "Annual Equipment Failures",
    # "Tenure":                "Customer Tenure",
    "MonthlyCharge": "Average Monthly Charge",
    "Bandwidth_GB_Year": "Annual Bandwidth Usage",
}

for column, variable_name in numerical_column_titles.items():
    # Setting up the figure and axes for side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    plt.suptitle(f"Exploration of {variable_name}")

    # Plot 1: Histogram of Age on ax1
    sns.histplot(df[column], bins=30, kde=True, color="green", ax=ax1)
    ax1.set_title(f"Distribution of {variable_name}")
    ax1.set_xlabel(variable_name)
    ax1.set_ylabel("Frequency")
    ax1.grid(True)

    # Plot 2: Scatter Plot of Age vs. Tenure on ax2 using regplot for a potential regression line
    sns.regplot(
        x=column,
        y="Tenure",
        data=df,
        color="green",
        ax=ax2,
        scatter_kws={"alpha": 1 / 10},
    )
    ax2.set_title(f"{variable_name} vs. Customer Tenure")
    ax2.set_xlabel(variable_name)
    ax2.set_ylabel("Tenure (months)")
    ax2.grid(True)

    # Show the plots
    plt.tight_layout()  # Adjusts plot parameters to give some padding and prevent overlap
    plt.show()

categorical_column_titles = {
    "State": "Customer State of Residence",
    "Area": "Customer Area Type",
    "TimeZone": "Time Zone of Customer Residence",
    "Marital": "Marital Status of Customer",
    "Gender": "Gender of Customer",
    "Churn": "Customer Churn Status Last Month",
    "Techie": "Whether Customer is Tech-Savvy",
    "Contract": "Type of Customer Contract",
    "Port_modem": "Whether Customer Uses a Portable Modem",
    "Tablet": "Whether Customer Owns a Tablet",
    "InternetService": "Type of Internet Service Customer Uses",
    "Phone": "Whether Customer Has Phone Service",
    "Multiple": "Whether Customer Has Multiple Lines",
    "OnlineSecurity": "Whether Customer Uses Online Security Service",
    "OnlineBackup": "Whether Customer Uses Online Backup Service",
    "DeviceProtection": "Whether Customer Uses Device Protection Service",
    "TechSupport": "Whether Customer Has Technical Support Service",
    "StreamingTV": "Whether Customer Uses Streaming TV Service",
    "StreamingMovies": "Whether Customer Uses Streaming Movies Service",
    "PaperlessBilling": "Whether Customer Uses Paperless Billing",
    "PaymentMethod": "Customer's Payment Method",
}


for column, variable_name in categorical_column_titles.items():
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    area_counts = df[column].value_counts()
    colors = sns.color_palette(
        "Greens", n_colors=area_counts.size
    )  # Using green color palette
    ax[0].pie(
        area_counts,
        labels=area_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
    )
    ax[0].set_title(f"Distribution of {variable_name}")

    # Box plot
    sns.boxplot(x=column, y="Tenure", data=df, ax=ax[1], palette="Greens")
    ax[1].set_title(f"{variable_name} vs Tenure")
    ax[1].set_xlabel(variable_name)
    ax[1].set_ylabel("Tenure (months)")

    # Show the plot
    plt.tight_layout()
    plt.show()

input("Press enter to begin Part C4")
# C4: Data Transformation

# Convert zip codes to string to preserve leading zeros
df["Zip"] = (
    df["Zip"].astype(str).str.zfill(5)
)  # Assuming "zip" is the name of the column for zip codes

# Mapping of locations to time zones
time_zone_map = {
    "America/New_York": "EST",
    "America/Detroit": "EST",
    "America/Indiana/Indianapolis": "EST",
    "America/Kentucky/Louisville": "EST",
    "America/Indiana/Vincennes": "EST",
    "America/Indiana/Tell_City": "EST",
    "America/Indiana/Petersburg": "EST",
    "America/Indiana/Knox": "EST",
    "America/Indiana/Winamac": "EST",
    "America/Indiana/Marengo": "EST",
    "America/Toronto": "EST",
    "America/Chicago": "CST",
    "America/Menominee": "CST",
    "America/North_Dakota/New_Salem": "CST",
    "America/Denver": "MST",
    "America/Phoenix": "MST",
    "America/Boise": "MST",
    "America/Los_Angeles": "PST",
    "America/Anchorage": "AKST",
    "America/Nome": "AKST",
    "America/Sitka": "AKST",
    "America/Juneau": "AKST",
    "Pacific/Honolulu": "HAST",
    "America/Puerto_Rico": "AST",
    "America/Ojinaga": "MST",
}

# Replace the TimeZone column with the mapped values
df["TimeZone"] = df["TimeZone"].map(time_zone_map)

# Convert boolean columns to actual boolean types
boolean_columns = [
    "Techie",
    "Port_modem",
    "Tablet",
    "Phone",
    "Multiple",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]
for column in boolean_columns:
    df[column] = df[column].map({"Yes": True, "No": False})

# Convert remaining categorical data to category dtype
nominal_columns = [
    "Area",
    "TimeZone",
    "Job",
    "Marital",
    "Gender",
    "Contract",
    "InternetService",
    "PaymentMethod",
]
for column in nominal_columns:
    df[column] = df[column].astype("category")

# Create a new dataframe with only relevant variables
df_encoded = df[
    [
        "Population",
        "Area",
        "TimeZone",
        "Children",
        "Age",
        "Income",
        "Marital",
        "Gender",
        "Outage_sec_perweek",
        "Email",
        "Contacts",
        "Yearly_equip_failure",
        "Techie",
        "Contract",
        "Port_modem",
        "Tablet",
        "InternetService",
        "Phone",
        "Multiple",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "PaymentMethod",
        "Tenure",
        "MonthlyCharge",
        "Bandwidth_GB_Year",
    ]
].copy()
df_encoded = pd.get_dummies(
    df_encoded,
    columns=[
        "Area",
        "Gender",
        "Contract",
        "Marital",
        "TimeZone",
        "InternetService",
        "PaymentMethod",
    ],
    drop_first = True,
)

print(df_encoded)

df_encoded.to_csv("churn_encoded.csv")

input("Press enter to begin Part D1")
# D1: Initial Linear Regression Model

Y = df_encoded["Tenure"]
X = df_encoded.drop(columns=["Tenure"])
X = sm.add_constant(X)
model = sm.OLS(Y, X.astype(float))
results = model.fit()
print(results.summary())

input("Press enter to begin Part D2")
# D2: Reduced Feature Set
X = df_encoded.drop(columns=["Tenure"])
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

vif_data["VIF"] = [
    variance_inflation_factor(X.values.astype(float), i) for i in range(len(X.columns))
]

print(vif_data)

# Analysis with MonthlyCharge removed
X = df_encoded.drop(columns=["Tenure", "MonthlyCharge"])
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

vif_data["VIF"] = [
    variance_inflation_factor(X.values.astype(float), i) for i in range(len(X.columns))
]

print(vif_data)

# Apply Backward Elimination
Y = df_encoded["Tenure"]
columns_to_drop = [
    "Area_Urban",
    "Gender_Nonbinary",
    "Contract_Two Year",
    "Marital_Widowed",
    "TimeZone_PST",
    "InternetService_None",
    "PaymentMethod_Mailed Check",
]
X = df_encoded.drop(columns=["Tenure", "MonthlyCharge"] + columns_to_drop)
X_optimal, model = backward_elimination(
    X.astype(float), df_encoded["Tenure"]
)

print(model.summary())

input("Press enter to begin Part D3")
# D3: Create a reduced model
model_reduced = sm.OLS(Y, X_optimal)
results_reduced = model_reduced.fit()
print(results_reduced.summary())

input("Press enter to begin Part E2")
# E2. Residual Plot and RSE
# Residual Plot for the Reduced Model
plt.figure(figsize=(12, 6))
plt.scatter(results_reduced.fittedvalues, results_reduced.resid)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residual Plot of the Reduced Model")
plt.show()

# Calculate and print the Residual Standard Error
RSE = np.sqrt(results_reduced.scale)
print(f"Residual Standard Error (RSE) of the Reduced Model: {RSE:.3f}")
