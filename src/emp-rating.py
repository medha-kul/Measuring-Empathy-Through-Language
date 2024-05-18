
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F
# Load the empathy dataset
empathy_df = pd.read_csv('./emp-rating/EmpathyDataset.csv')
# Load the CounselChat dataset
counsel_chat_df = pd.read_csv('./counsel_chat.csv')
# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to encode the dataset
def tokenize_and_encode(statements):
    return tokenizer(statements, padding="max_length", truncation=True, max_length=128)

# Split the empathy data
train_texts, test_texts, train_labels, test_labels = train_test_split(empathy_df['Statement'], empathy_df['Rating'], test_size=0.2)

# Encode texts
train_encodings = tokenize_and_encode(train_texts.tolist())
test_encodings = tokenize_and_encode(test_texts.tolist())

# Custom dataset class for PyTorch
class EmpathyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset objects
train_dataset = EmpathyDataset(train_encodings, train_labels.tolist())
test_dataset = EmpathyDataset(test_encodings, test_labels.tolist())

# Load BERT with a classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

def predict_empathy(texts):
    # Encode the texts
    encoded_texts = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**encoded_texts)
        predictions = F.softmax(outputs.logits, dim=-1)[:, 1]  # Get the probability of '1' (Empathic)
    
    return predictions.cpu().numpy()



# Apply the empathy model to the 	answers in the CounselChat dataset
counsel_chat_df['EmpathyRating'] = counsel_chat_df['answerText'].apply(lambda x: predict_empathy([x])[0])

# Save the updated dataset
counsel_chat_df.to_csv('./updated_counsel_chat.csv', index=False)

# Example usage
sample_texts = [
    "I can understand how challenging this situation must be for you, and I'm here to help you through it.",
    "I'm here for you, and I appreciate you sharing this with me. Let's figure out a way forward together.",
    "Why don't you just get over it? Everyone goes through tough times.",
    "That doesn't sound like a big deal. You shouldn't let it bother you so much."
]
predictions = predict_empathy(sample_texts)

for text, pred in zip(sample_texts, predictions):
    print(f"Text: {text}\nPrediction: {'Empathic' if pred >0.5 else 'Not Empathic'}\n")
