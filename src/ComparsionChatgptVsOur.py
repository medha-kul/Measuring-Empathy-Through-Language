import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
chatgpt_data = pd.read_csv('./chatgpt_counsel_chat.csv')
updated_data = pd.read_csv('./updated_counsel_chat.csv')

# Merge the datasets on 'questionID'
merged_data = pd.merge(chatgpt_data, updated_data[['questionID', 'Overall Empathy Score']],
                       on='questionID', how='inner')

# Calculate average scores for each dataset
average_chatgpt_score = merged_data['empathyScore'].mean()
average_updated_score = merged_data['Overall Empathy Score'].mean()

# Plot the average scores
plt.figure(figsize=(10, 5))
bar_locations = [1, 2]
averages = [average_chatgpt_score, average_updated_score]
labels = ['ChatGPT Empathy Score', 'Updated Overall Empathy Score']
plt.bar(bar_locations, averages, tick_label=labels, color=['blue', 'orange'])
plt.title('Average Empathy Scores Comparison')
plt.ylabel('Average Score')
plt.ylim(0, 1)  # Assuming scores are normalized between 0 and 1
for i, v in enumerate(averages):
    plt.text(i + 1, v + 0.02, f"{v:.2f}", ha='center')
plt.show()

# Plot histograms of both scores for distribution comparison
plt.figure(figsize=(10, 5))
plt.hist(merged_data['empathyScore'], bins=20, alpha=0.5, label='ChatGPT Empathy Score')
plt.hist(merged_data['Overall Empathy Score'], bins=20, alpha=0.5, label='Calculated Empathy Score')
plt.title('Distribution of Empathy Scores')
plt.xlabel('Empathy Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()
