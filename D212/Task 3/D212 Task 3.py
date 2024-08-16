#!/usr/bin/env python
# coding: utf-8

# DATA MINING II - D212
# 
# Task 3
# 
# Eric D. Otten
# 
# Student ID: 011399183
# 
# # A1.  Propose one question relevant to a real-world organizational situation that you will answer using market basket analysis.
# 
# My research question for D212 Task 3 is: "How can we identify key associations of customer purchases from the telecommunications company using its market basket dataset?" 

# # A2.  Define one goal of the data analysis. Ensure your goal is reasonable within the scope of the selected scenario and is represented in the available data.
# 
# My primary goal is to uncover the key associations between customer purchases to enhance our understanding of customer behavior. 
# 
# By leveraging market basket analysis, I aim to identify patterns and relationships among the products and services customers frequently buy together. This analysis will provide valuable insights into customer preferences and purchasing habits, enabling us to develop targeted marketing strategies and personalized discount offers.
# 
# Ultimately, these findings will help improve customer satisfaction and loyalty, ensuring that the telecommunication company's offerings align more closely with customer needs and preferences. This goal is both reasonable and attainable, given the rich data available in our market basket dataset, which captures detailed transactional information across a diverse customer base.

# # B1. Explain the reasons for using market basket analysis by doing the following: 1.  Explain how market basket analyzes the selected data set. Include expected outcomes.
# 
# Market basket analysis is a powerful tool for understanding customer purchasing patterns by examining the items they buy together. For our telecommunications company, it involves analyzing the 7,501 customer purchase histories, each consisting of up to 20 different items related to technological areas.
# 
# This method uses association rules to identify relationships between items, which can then highlight frequently co-purchased products. By examining these associations, we can uncover trends and patterns that may not be immediately obvious. The expected outcomes include discovering which products are often bought together, identifying complementary products, and understanding customer preferences more deeply. 
# 
# Imagine, if you will, a customer's purchase history includes items such as a smartphone, a Bluetooth headset, a screen protector, and a charging cable.
# 
# Using market basket analysis, we can identify that customers who buy smartphones are also likely to purchase screen protectors and Bluetooth headsets. This information allows us to create promotional bundles or targeted marketing campaigns that highlight these commonly co-purchased items, thereby increasing the likelihood of additional sales and improving the customer experience.

# # B2.  Provide one example of transactions in the data set.
# 
# The code below performs some necessary data preparation such as removing empty rows before identifying the 17th (index: 16) transaction.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("teleco_market_basket.csv")

df = df[df["Item01"].notna()]
df.reset_index(drop=True, inplace=True)
print(df.iloc[16])


# # B3.  Summarize one assumption of market basket analysis.
# 
# One key assumption of market basket analysis is that the purchasing behavior of customers in the dataset is representative of the broader customer base. This means we assume that the patterns and associations identified from the historical purchase data of the 7,501 customers are indicative of the general purchasing behavior of all our customers.
# 
# This assumption is crucial because it underpins the validity of the insights and recommendations derived from the analysis. By ensuring our data is comprehensive and accurately reflects customer behavior, we can confidently use these insights to inform our marketing and sales strategies, ultimately driving better business outcomes.

# # C1. Prepare and perform market basket analysis by doing the following: 1.  Transform the data set to make it suitable for market basket analysis. Include a copy of the cleaned data set.
# 

# In[2]:


# Convert df into lists
transactions = []
for i in range(len(df)):
    transaction = [str(df.values[i, j]) for j in range(20) if str(df.values[i, j]) != "nan"]
    transactions.append(transaction)

te = TransactionEncoder()

te_array = te.fit(transactions).transform(transactions)
encoded_df = pd.DataFrame(te_array, columns=te.columns_)
print(encoded_df)


# In[3]:


frequent_itemsets = apriori(encoded_df, min_support = 0.02, use_colnames = True)
print(frequent_itemsets)


# # C2.  Execute the code used to generate association rules with the Apriori algorithm. Provide screenshots that demonstrate that the code is error free.
# 

# In[4]:


rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1.0)
print(rules)


# # C3.  Provide values for the support, lift, and confidence of the association rules table.
# 

# In[5]:


rules


# # C4.  Explain the top three relevant rules generated by the Apriori algorithm. Include a screenshot of the top three relevant rules.
# 

# In[7]:


top_3_rules = rules.sort_values(by="lift", ascending=False).head(3)
print(top_3_rules)


# # D1. Summarize your data analysis by doing the following: Summarize the significance of support, lift, and confidence from the results of the analysis.
# 
# Support: The support metric tells us how often an itemset appears in the dataset. For instance, the rule involving the VIVO Dual LCD Monitor Desk Mount and SanDisk Ultra 64GB card has a support of 0.039195, indicating these items are purchased together in approximately 3.92% of all transactions. This suggests a modestly frequent occurrence worth noting for marketing strategies.
# 
# Confidence: Confidence measures the likelihood that the consequent is bought when the antecedent is purchased. The confidence of 22.51% in the first rule means that there is a 22.51% chance of selling a SanDisk Ultra 64GB card when a VIVO Dual LCD Monitor Desk Mount is purchased. The second rule shows a higher confidence of 39.89%, indicating a stronger likelihood of the desk mount being purchased when customers buy the memory card.
# 
# Lift: Lift indicates the strength of the association between items beyond coincidence. A lift value greater than 1, such as 2.29 in these rules, suggests a positive association. This high lift value reveals that customers are more than twice as likely to purchase these items together compared to buying them independently.

# # D2.  Discuss the practical significance of your findings from the analysis.
# 
# The observed relationships suggest that customers interested in office or tech products like monitor mounts and memory cards exhibit specific bundled purchasing behaviors.
# 
# This insight allows for strategic product placement and promotional bundling, which could enhance the shopping experience and increase sales. Furthermore, the association between FEIYOLD Blue light Blocking Glasses and the VIVO Dual LCD Monitor Desk Mount with a lift of nearly 2.00 indicates a significant cross-selling opportunity, especially among customers looking to optimize their workstation setups.

# # D3.  Recommend a course of action for the real-world organizational situation from part A1 based on the results from part D1.
# 
# Given these insights, I recommend the following actions for the telecommunications company:
# 
# Bundle the VIVO Dual LCD Monitor Desk Mount with SanDisk Ultra 64GB cards and possibly include FEIYOLD Blue light Blocking Glasses in a special offer. This could attract more customers looking to purchase comprehensive workstation setups.
# 
# Use the data from these rules to target advertisements more precisely. Customers buying any of these products could be shown ads for the other items, potentially in real-time as they shop online or through follow-up marketing campaigns.
# 
# Optimize in person and e-commerce store layouts. In physical stores, placing these items near each other can encourage impulse buys, capitalizing on the revealed purchasing patterns.

# # F. Web Sources
# 
# TransactionEncoder docs - https://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/

# # G. Works Consulted
