from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

df = pd.read_csv("churn_clean.csv", keep_default_na=False, na_values=["NA"])

# B1: Data Set Analysis
table = pd.crosstab(df.Tenure, df.Marital)
grouped_tenures = df.groupby("Marital")["Tenure"].apply(list).to_dict()

tenures_married = grouped_tenures["Married"]
tenures_separated = grouped_tenures["Separated"]
tenures_divorced = grouped_tenures["Divorced"]
tenures_widowed = grouped_tenures["Widowed"]
tenures_never_married = grouped_tenures["Never Married"]

anova_result = f_oneway(
    tenures_married,
    tenures_separated,
    tenures_divorced,
    tenures_widowed,
    tenures_never_married,
)

print("F-statistic:", anova_result.statistic)
print("P-value:", anova_result.pvalue)


# C. Distribution Using Univariate Statistics
# Calculate basic statistics for 'Tenure' and 'Income'
tenure_stats = df['Tenure'].describe()
income_stats = df['Income'].describe()

print("Tenure Statistics:")
print(tenure_stats)

print("\nIncome Statistics:")
print(income_stats)

# Calculate frequencies and proportions for 'Marital' and 'Area'
marital_counts = df['Marital'].value_counts()
marital_proportions = df['Marital'].value_counts(normalize=True)

area_counts = df['Area'].value_counts()
area_proportions = df['Area'].value_counts(normalize=True)

print("\nMarital Status Distribution:")
print(marital_counts)
print(marital_proportions)

print("\nArea Distribution:")
print(area_counts)
print(area_proportions)

# C1. Visual of Findings
# Histogram for Tenure
plt.figure(figsize=(8, 6))
plt.hist(df["Tenure"], bins=30, color="skyblue", edgecolor="black")
plt.title("Histogram of Tenure")
plt.xlabel("Tenure (years)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Histogram for Income
plt.figure(figsize=(8, 6))
plt.hist(df["Income"], bins=30, color="green", edgecolor="black")
plt.title("Histogram of Income")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Pie Chart for Marital Status
marital_counts = df["Marital"].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(
    marital_counts,
    labels=marital_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=["gold", "lightblue", "lightgreen", "salmon", "purple"],
)
plt.title("Pie Chart of Marital Status")
plt.show()

# Pie Chart for Area
area_counts = df["Area"].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(
    area_counts,
    labels=area_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=["lavender", "teal", "coral"],
)
plt.title("Pie Chart of Area")
plt.show()




# D. Distribution Using Bivariate Statistics
# Calculate correlation coefficient for Income and MonthlyCharge
income_monthly_charge_corr = df['Income'].corr(df['MonthlyCharge'])
print(f"Correlation coefficient between Income and Monthly Charge: {income_monthly_charge_corr:.2f}")

# Calculate correlation coefficient for Tenure and Bandwidth_GB_Year
tenure_bandwidth_corr = df['Tenure'].corr(df['Bandwidth_GB_Year'])
print(f"Correlation coefficient between Tenure and Bandwidth_GB_Year: {tenure_bandwidth_corr:.2f}")

# Contingency Table for Internet Service vs Contract Type
internet_contract_table = pd.crosstab(df['InternetService'], df['Contract'])
print("\nContingency Table for Internet Service vs Contract Type:")
print(internet_contract_table)

# Perform Chi-squared test
chi2_stat, p_val, dof, expected = chi2_contingency(internet_contract_table)
print(f"\nChi-squared Test results for Internet Service vs Contract Type:")
print(f"Chi-squared Statistic: {chi2_stat}, P-value: {p_val}")

# Contingency Table for Churn vs Payment Method
churn_payment_table = pd.crosstab(df['Churn'], df['PaymentMethod'])
print("\nContingency Table for Churn vs Payment Method:")
print(churn_payment_table)

# Perform Chi-squared test
chi2_stat, p_val, dof, expected = chi2_contingency(churn_payment_table)
print(f"\nChi-squared Test results for Churn vs Payment Method:")
print(f"Chi-squared Statistic: {chi2_stat}, P-value: {p_val}")


# D1. Visual of Findings
# Plot for Income vs. MonthlyCharge
plt.figure(figsize=(10, 6))
sns.regplot(x="Income", y="MonthlyCharge", data=df, scatter_kws={"alpha": 0.5})
plt.title("Linear Regression: Income vs Monthly Charge")
plt.xlabel("Income")
plt.ylabel("Monthly Charge")
plt.show()

# Plot for Tenure vs. Bandwidth_GB_Year
plt.figure(figsize=(10, 6))
sns.regplot(x="Tenure", y="Bandwidth_GB_Year", data=df, scatter_kws={"alpha": 0.5})
plt.title("Linear Regression: Tenure vs Bandwidth GB Year")
plt.xlabel("Tenure (years)")
plt.ylabel("Bandwidth GB Year")
plt.show()

# Data preparation for InternetService vs. Contract
internet_contract = pd.crosstab(df['InternetService'], df['Contract'])
internet_contract.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Stacked Bar Chart: Internet Service vs Contract')
plt.xlabel('Internet Service')
plt.ylabel('Count')
plt.show()

# Data preparation for Churn vs. PaymentMethod
churn_payment = pd.crosstab(df['Churn'], df['PaymentMethod'])
churn_payment.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Stacked Bar Chart: Churn vs Payment Method')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()