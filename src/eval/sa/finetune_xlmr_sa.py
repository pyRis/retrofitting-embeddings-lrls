import os
import argparse
import pandas as pd
import numpy as np
import string
import torch
from nltk import word_tokenize
import nltk
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from huggingface_hub import HfApi, login

nltk.download('punkt')

# Define arguments
parser = argparse.ArgumentParser(description='Script for fine-tuning XLM-R model for classification')
parser.add_argument('--language', type=str, required=True, help='Language code for the dataset')
parser.add_argument('--huggingface_token', type=str, required=True, help='Hugging Face token for authentication')
parser.add_argument('--repo_name', type=str, required=True, help='Name of the Hugging Face repository')


args = parser.parse_args()

# Define language
language = args.language
languages_mapping = {"ro": "romanian", "da": "danish", "he": "hebrew", 
                     "sl": "slovenian", "lv": "latvian", "th": "thai", 
                     "ur": "urdu", "cy": "welsh", "az": 'azerbaijani', 
                     "el": 'greek', "sk": 'slovak', "ka": 'georgian', 
                     "bn": 'bengali', "mk": 'macedonian','ku': 'kurdish', 
                     'te': 'telugu', 'mr': 'marathi', 'uz': "uzbek", 
                     'sw': 'swahili', 'yo': 'yoruba', 'ug': "uyghur", 
                     'ne': 'nepali', 'jv': 'javanese', 'si': 'sinhala', 
                     'su': 'sundanese', 'bg': 'bulgarian', 'am': 'amharic'}

# Load dataset
dataset = load_dataset(f"DGurgurov/{languages_mapping[language]}_sa")

# Define function for text preprocessing
def preprocess_text(text):
    """Function that preprocesses text by removing punct, lowercasing and tokenizing"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    text = ' '.join(words)
    return text

# Prepare data
train = pd.DataFrame(dataset["train"])
valid = pd.DataFrame(dataset["validation"])
test = pd.DataFrame(dataset["test"])

train['text'] = train['text'].apply(preprocess_text)
valid['text'] = valid['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')

def tokenize_data(texts, labels):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True)
    return dict(encodings, labels=labels.tolist())

train_encodings = tokenize_data(train['text'], train['label'])
valid_encodings = tokenize_data(valid['text'], valid['label'])
test_encodings = tokenize_data(test['text'], test['label'])

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = SentimentDataset(train_encodings)
valid_dataset = SentimentDataset(valid_encodings)
test_dataset = SentimentDataset(test_encodings)

# Load model
model = AutoModelForSequenceClassification.from_pretrained('FacebookAI/xlm-roberta-base', num_labels=2).to('cuda')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results_xlm_r',
    num_train_epochs=20,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_xlm_r',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Define metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_score(p.label_ids, preds)
    macro_f1 = f1_score(p.label_ids, preds, average='macro')
    micro_f1 = f1_score(p.label_ids, preds, average='micro')
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate model
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)

print("Fine-tuned XLM-R:")
print(classification_report(test['label'], preds, digits=3))

def save_results(file_path, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    conf_matrix = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:
        file.write(f"Accuracy: {accuracy:.5f}\n")
        file.write(f"Macro Average F1 Score: {macro_f1:.5f}\n")
        file.write(f"Micro Average F1 Score: {micro_f1:.5f}\n")
        file.write("Confusion Matrix:\n")
        file.write(np.array2string(conf_matrix, separator=', '))

save_results(f'xlm-r-base-ft/{language}/fine_tuned_xlm-r.txt', test['label'], preds)

# Save model and tokenizer
model.save_pretrained(f'xlm-r-base-ft/{language}')
tokenizer.save_pretrained(f'xlm-r-base-ft/{language}')

# Login to Hugging Face Hub
login(token=args.huggingface_token)

# Initialize HfApi
api = HfApi()

# Create a new repository on Hugging Face Hub if it doesn't exist
repo_name = args.repo_name
user = api.whoami()['name']
repo_id = f"{user}/{repo_name}_{languages_mapping[language]}_sentiment"
api.create_repo(repo_id=repo_id, exist_ok=True)

# Upload model and tokenizer to the repository
api.upload_folder(
    folder_path=f'xlm-r-base-ft/{language}',
    path_in_repo='',
    repo_id=repo_id
)

# Create a README.md file with model details
readme_content = f"""
# Fine-tuned XLM-R Model for {languages_mapping[language]} Sentiment Analysis

This is a fine-tuned XLM-R model for sentiment analysis in {languages_mapping[language]}.

## Model Details

- **Model Name**: XLM-R Sentiment Analysis
- **Language**: {languages_mapping[language]}
- **Fine-tuning Dataset**: DGurgurov/{languages_mapping[language]}_sa

## Training Details

- **Epochs**: 20
- **Batch Size**: 32 (train), 64 (eval)
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5

## Performance Metrics

- **Accuracy**: {accuracy_score(test['label'], preds):.5f}
- **Macro F1**: {f1_score(test['label'], preds, average='macro'):.5f}
- **Micro F1**: {f1_score(test['label'], preds, average='micro'):.5f}

## Usage

To use this model, you can load it with the Hugging Face Transformers library:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")
```

## License
[MIT]
"""

with open(f'xlm-r-base-ft/{language}/README.md', 'w') as f:
    f.write(readme_content)


api.upload_file(
path_or_fileobj=f'xlm-r-base-ft/{language}/README.md',
path_in_repo='README.md',
repo_id=repo_id
)
