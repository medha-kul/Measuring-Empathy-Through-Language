import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('./updated_counsel_chat.csv')

# Display the first few rows and column names to understand the dataset structure
print(df.head())
print(df.columns)

# Scatter plot for Empathy Scores vs. Upvotes
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Overall Empathy Score', y='upvotes', data=df)
plt.title('Scatter Plot of Empathy Scores vs. Upvotes')
plt.xlabel('Overall Empathy Score')
plt.ylabel('Upvotes')
plt.grid(True)
plt.show()

# Scatter plot for Empathy Scores vs. Views
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Overall Empathy Score', y='views', data=df)
plt.title('Scatter Plot of Empathy Scores vs. Views')
plt.xlabel('Overall Empathy Score')
plt.ylabel('Views')
plt.grid(True)
plt.show()

# Histogram of Empathy Scores
plt.figure(figsize=(10, 6))
sns.histplot(df['Overall Empathy Score'], kde=True)
plt.title('Distribution of Empathy Scores')
plt.xlabel('Overall Empathy Score')
plt.ylabel('Frequency')
plt.show()
