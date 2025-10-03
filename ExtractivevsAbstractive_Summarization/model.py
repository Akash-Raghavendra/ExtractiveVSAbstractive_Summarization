import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch

# File paths for the train, validation and test sets
train_path = 'samsum-train.csv'  
validation_path = 'samsum-validation.csv'  
test_path = 'samsum-test.csv'  

# Loading dataset using pandas
train_df = pd.read_csv(train_path)
validate_df = pd.read_csv(validation_path)
test_df = pd.read_csv(test_path)

# Initializing the T5 transformer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Preprocessing function 
def preprocess_text_for_t5(text):
    # Converting non-string inputs to string and handle missing data
    text = str(text) if text is not None else ""  # if text is empty then none
    return tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

# Applying the preprocessing to dialogues and summaries for all datasets
train_df['dialogue_tokenized'] = train_df['dialogue'].apply(lambda x: preprocess_text_for_t5(x)['input_ids'].squeeze())
train_df['summary_tokenized'] = train_df['summary'].apply(lambda x: preprocess_text_for_t5(x)['input_ids'].squeeze())

validate_df['dialogue_tokenized'] = validate_df['dialogue'].apply(lambda x: preprocess_text_for_t5(x)['input_ids'].squeeze())
validate_df['summary_tokenized'] = validate_df['summary'].apply(lambda x: preprocess_text_for_t5(x)['input_ids'].squeeze())

test_df['dialogue_tokenized'] = test_df['dialogue'].apply(lambda x: preprocess_text_for_t5(x)['input_ids'].squeeze())
test_df['summary_tokenized'] = test_df['summary'].apply(lambda x: preprocess_text_for_t5(x)['input_ids'].squeeze())

# Preview of processed train dataset
print(train_df.head())

# Converting data to dataset format for training
class TextSummarizationDataset:
    def __init__(self, dialogues, summaries):
        self.dialogues = dialogues
        self.summaries = summaries

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        summary = self.summaries[idx]

        # No additional padding is required as padding is already handled by the tokenizer
        return {
            'input_ids': dialogue,
            'labels': summary
        }

# Preparing datasets
train_dataset = TextSummarizationDataset(train_df['dialogue_tokenized'].values, train_df['summary_tokenized'].values)
val_dataset = TextSummarizationDataset(validate_df['dialogue_tokenized'].values, validate_df['summary_tokenized'].values)
test_dataset = TextSummarizationDataset(test_df['dialogue_tokenized'].values, test_df['summary_tokenized'].values)

# Define training arguments 
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    learning_rate=2e-5,              # learning rate for optimizer
    per_device_train_batch_size=4,   # batch size for training
    per_device_eval_batch_size=4,    # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    save_total_limit=3,              # limit the number of saved checkpoints
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,                # how often to log
)

# Initializing the T5 model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Initializing the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Training the model
trainer.train()

# Saving the model and tokenizer
model.save_pretrained('./summarization_model')
tokenizer.save_pretrained('./summarization_model')
