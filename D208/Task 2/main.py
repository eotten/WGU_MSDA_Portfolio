from matplotlib.colors import ListedColormap
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score


def to_snake_case(df):
    new_df = df.copy()

    def convert(name):
        # Handle the internal capital letters and add an underscore before capitals
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        # Replace spaces and other special characters with underscores
        name = re.sub(r'\s+|\W', '_', name)
        # Remove any double underscores caused by previous replacements
        name = re.sub(r'__+', '_', name)
        # Remove any trailing underscore
        name = re.sub(r'_+$', '', name)
        return name

    # Apply the convert function to each column name
    new_df.columns = [convert(col) for col in new_df.columns]
    return new_df

# Path to the CSV file
file_path = 'churn_clean.csv'

# Load the entire dataset
df = pd.read_csv(file_path, keep_default_na = False, na_values=["NA"])



input("Press Enter to begin Part C1...")
# C1. Summarize the data preparation process for logistic regression by doing the following: Describe your data cleaning goals and the steps used to clean the data to achieve the goals that align with your research question including the annotated code.

# List of columns to select
selected_columns = [
    'Area', 'Children', 'Age', 'Income', 'Marital', 'Gender', 'Contract',
    'Port_modem', 'Tablet', 'InternetService', 'Phone', 'Multiple',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod',
    'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Outage_sec_perweek',
    'Email', 'Contacts', 'Yearly_equip_failure', 'Techie', 'Item1', 'Item2',
    'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8', 'Churn'
]

# Select the specified columns
df = df[selected_columns]

# Rename all columns using snake_case
df = to_snake_case(df)

# Columns that are expected to be boolean
boolean_columns = [
    'port_modem', 'tablet', 'phone', 'multiple', 'online_security', 'online_backup', 
    'device_protection', 'tech_support', 'streaming_t_v', 'streaming_movies', 
    'paperless_billing', 'techie', 'churn'
]

# Convert "yes" to True and "no" to False
for column in boolean_columns:
    df[column] = df[column].map({'Yes': True, 'No': False})

# Print summary statistics before creating dummy variables
categorical_columns = [
    'marital',
    'area',
    'gender',
    'contract',
    'internet_service',
    'payment_method',
]
# Convert specified columns to 'category' dtype
for column in categorical_columns:
    df[column] = df[column].astype('category')





input("Press Enter to begin Part C2...")
# C2. Describe the dependent variable and all independent variables using summary statistics that are required to answer the research question, including a screenshot of the summary statistics output for each of these variables.

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Generate summary statistics for numerical data
for column in numerical_columns.drop(["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8"]):
    print(f"\nSummary statistics for column: {column}")
    print(df[column].describe())

print("\nFrequency Distribution for Categorical Data:")
for column in categorical_columns:
    print(f"\nFrequency count for column: {column}")
    print(df[column].value_counts())


input("Press Enter to begin Part C3...")
# C3. Generate univariate and bivariate visualizations of the distributions of the dependent and independent variables, including the dependent variable in your bivariate visualizations.
palette = "rocket_r"
for column in numerical_columns:
    with sns.axes_style("whitegrid"):
        # Create a figure for each column with 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram on the left
        histplot = sns.histplot(df[column], kde=False, ax=axes[0])
        axes[0].set_title(f'Distribution of {column}')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Frequency')

        cm = sns.color_palette(palette, len(histplot.patches))
        for bin_, i in zip(histplot.patches, cm):
            bin_.set_facecolor(i)

        # Box plot on the right comparing with 'Churn'
        sns.boxplot(x='churn', y=column, data=df, ax=axes[1], palette=palette)
        axes[1].set_title(f'{column} vs churn')
        axes[1].set_xlabel('churn')
        axes[1].set_ylabel(column)

        # Display the plot
        plt.tight_layout()
        plt.show()

palette = "crest"
for column in (categorical_columns + boolean_columns):
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart on the left
        df[column].value_counts().plot.pie(ax=axes[0], autopct='%1.1f%%', startangle=90, colormap=palette, explode=[0.1]*df[column].nunique())
        axes[0].set_title(f'Distribution of {column}')
        axes[0].set_ylabel('')  # Hide the y-label

        # Contingency table on the right
        contingency_table = pd.crosstab(df[column], df['churn'])
        sns.heatmap(contingency_table, annot=True, fmt="d", cmap=palette, ax=axes[1], cbar=False)
        axes[1].set_title(f'{column} vs churn')
        axes[1].set_xlabel('churn')
        axes[1].set_ylabel(column)

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

input("Press Enter to begin Part C4...")
# C4. Describe your data transformation goals that align with your research question and the steps used to transform the data to achieve the goals, including the annotated code.

# Create dummy variables for specified columns and drop the first category of each
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Rename all columns using snake_case
df = to_snake_case(df)

print(df.head())
df.to_csv("churn_encoded.csv")


input("Press Enter to begin Part D1...")
# D1. Compare an initial and a reduced logistic regression model by doing the following: Construct an initial logistic regression model from all independent variables that were identified in part C2.
X = df.drop(columns=['churn']).astype(int)
y = df['churn']

X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 23)

initial_model = sm.Logit(y_train, X_train).fit()

y_pred = initial_model.predict(X_test)
predicted_classes = (y_pred > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, predicted_classes)
precision = precision_score(y_test, predicted_classes)
recall = recall_score(y_test, predicted_classes)
conf_matrix = confusion_matrix(y_test, predicted_classes)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print('Confusion Matrix:')
print(conf_matrix)

print(initial_model.summary())

# Save training and test variables to CSV
X_train.to_csv('churn_X_train.csv')
X_test.to_csv('churn_X_test.csv')
y_train.to_csv('churn_y_train.csv')
y_test.to_csv('churn_y_test.csv')



input("Press Enter to begin Part D2...")
# D2. Justify a statistically based feature selection procedure or a model evaluation metric to reduce the initial model in a way that aligns with the research question.

target = "churn"
significance_level = 0.05 
X_reduced = X_train

round_count = 0
while True:
    round_count += 1
    print(f"\nRound {round_count} of backward elimination:")

    # Fit the model
    model = sm.Logit(y_train, X_reduced).fit()
    p_values = model.pvalues
    print(p_values)

    if p_values.max() > significance_level:
        feature_to_remove = p_values.idxmax()

        if feature_to_remove == "const":
            break

        print("Removing feature:", feature_to_remove)
        X_reduced = X_reduced.drop(columns=[feature_to_remove])
    else:
        break

print(model.summary())


input("Press Enter to begin Part D3...")
# D3. Provide a reduced logistic regression model that follows the feature selection or model evaluation process in part D2, including a screenshot of the output for each model.

# Evaluate the final model
X_test_reduced = X_test[X_reduced.columns] # X_test need to be adjusted to match the features selected in the reduced model.
y_pred = model.predict(X_test_reduced)
predicted_classes = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, predicted_classes)
precision = precision_score(y_test, predicted_classes)
recall = recall_score(y_test, predicted_classes)
conf_matrix = confusion_matrix(y_test, predicted_classes)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print('Confusion Matrix:')
print(conf_matrix)