import pandas as pd

# Load the dataset
path='./updated_counsel_chat.csv'

df = pd.read_csv(path)

# Normalize word count
df['Normalized Word Count'] = df['Word Count'] / df['Word Count'].max()

# Assigning numerical values to sentiment categories
def sentiment_value(sentiment):
    if sentiment == 'POSITIVE':
        return 0.3  # example weight, adjust as necessary
    elif sentiment == 'NEGATIVE':
        return -0.1  # example weight, can also be positive if that suits the context
    else:
        return 0  # for NEUTRAL or other undefined sentiments

df['Sentiment Value'] = df['Sentiment'].apply(sentiment_value)

# Define weights
w1, w2, w3 = 0.6, 0.3, 0.1  # Adjust these weights as per the importance you place on each factor

# Calculate overall empathy score
df['Overall Empathy Score'] = w1 * df['EmpathyRating'] + w2 * df['Sentiment'] + w3 * df['Normalized Word Count']

# Save or display the updated DataFrame
df.to_csv(path, index=False)

