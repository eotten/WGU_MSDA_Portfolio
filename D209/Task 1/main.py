import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.neighbors import KNeighborsRegressor

filename = "churn_clean.csv"
df = pd.read_csv(filename, keep_default_na=False, na_values=["NA"])

# C3: Data Cleaning

# Remove unused columns
df = df[[
    "Population", "Area", "TimeZone", "Children", "Age", "Income", "Marital", "Gender", "Churn",
    "Outage_sec_perweek", "Email", "Contacts", "Yearly_equip_failure", "Techie",
    "Contract", "Port_modem", "Tablet", "InternetService", "Phone", "Multiple",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "PaperlessBilling", "PaymentMethod", "Tenure", "MonthlyCharge",
    "Bandwidth_GB_Year", "Item1", "Item2", "Item3", "Item4", "Item5", "Item6", "Item7", "Item8"
]]

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
    "Churn",
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
    "Marital",
    "Gender",
    "Contract",
    "InternetService",
    "PaymentMethod",
]
for column in nominal_columns:
    df[column] = df[column].astype("category")

# Get dummy columns
df = pd.get_dummies(df, columns = nominal_columns, drop_first = True)

df.to_csv("churn_cleaned_final.csv")

# Print summary statistics
# Temporarily include 'CaseOrder' and 'Zip' as categorical variables
categorical_cols = df.select_dtypes(include=["object", "bool"]).copy()

# Update numerical columns to exclude 'CaseOrder' and 'Zip' if they were included
numerical_cols = df.select_dtypes(include=["int64", "float64"])


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


# D1.  Split the data into training and test data sets and provide the file(s).
df_X = df.drop(["Churn"], axis=1).copy()
df_y = df["Churn"].copy()

X = df_X.assign(const=1)
y = df_y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1111)

# Save the training and testing datasets to CSV files for reference
X_train.to_csv("churn_Xtrain.csv", index = False)
X_test.to_csv("churn_Xtest.csv", index = False)
y_train.to_csv("churn_ytrain.csv", index = False)
y_test.to_csv("churn_ytest.csv", index = False)


# D2.  Describe the analysis technique you used to appropriately analyze the data. Include screenshots of the intermediate calculations you performed.

# Determine the ideal value for k
parameters = {"n_neighbors": range(1, 50)}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)

print(gridsearch.best_params_)
print(gridsearch.best_score_)

# {"n_neighbors": 28}
# 0.1687078480616308

# Perform KNN using the value of k = 28
knn = KNeighborsClassifier(n_neighbors = 28)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Create a confusion matrix to compare the test (actual) values to the predicted values
cm = confusion_matrix(y_test, y_pred)

# Create a DataFrame from the matrix for readability
df_cm = pd.DataFrame(cm,
                     index = [label for label in ["Predicted No Churn", "Predicted Churn"]],
                     columns = [label for label in ["Actual No Churn", "Actual Churn"]])

print(df_cm)

# Extract FP and FN
total = cm.sum()
fp = cm[0][1]
fp_percent = fp / total * 100
fn = cm[1][0]
fn_percent = fn / total * 100

print("False Positives (FP):", fp, f"({fp_percent}%)")
print("False Negatives (FN):", fn, f"({fn_percent}%)")

# Plot the ROC curve
y_pred_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

# Calculate the area under the curve
auc = roc_auc_score(y_test, y_pred_prob)
print(f"Area under curve: {auc}")

# Generate a classification report
print(classification_report(y_test, y_pred))