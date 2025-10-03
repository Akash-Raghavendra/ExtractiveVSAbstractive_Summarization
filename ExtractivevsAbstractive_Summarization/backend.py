from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
import numpy as np

class Summarizer:
    def __init__(self, model_path, tokenizer_path):
        # Loading the trained model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    
    def generate_summary(self, input_text):
        # Tokenizing the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="longest")
        
        # Generating summary
        summary_ids = self.model.generate(inputs['input_ids'], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
        
        # Decoding the generated summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary

    def evaluate_rouge(self, reference_summary, generated_summary):
        # Initialize the Rouge scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        
        # Get the ROUGE scores
        scores = scorer.score(reference_summary, generated_summary)
        
        return scores

def extractive_summary_worse(input_text):
    # A worse-performing extractive summarization model using random sentence selection
    
    sentences = input_text.split('.')
    
    # Randomly select 3 sentences 
    selected_sentences = np.random.choice(sentences, size=3, replace=False)
    
    # Join the selected sentences to form the extractive summary
    extractive_summary = '. '.join(selected_sentences) + '.'
    
    return extractive_summary

def load_and_summarize(file_path):
    # Reading the file contents
    with open(file_path, 'r') as file:
        input_text = file.read()

    # Create the Summarizer instance
    summarizer = Summarizer('./summarization_model', './summarization_model')
    
    # Generate the abstractive summary using the trained model
    abstractive_summary = summarizer.generate_summary(input_text)
    
    # Generate the extractive summary 
    extractive_summary_text = extractive_summary_worse(input_text)
    
    # Evaluate ROUGE scores for both models
    reference_summary = input_text 
    abstractive_rouge = summarizer.evaluate_rouge(reference_summary, abstractive_summary)
    extractive_rouge = summarizer.evaluate_rouge(reference_summary, extractive_summary_text)

    # Return the summaries and the ROUGE scores
    return abstractive_summary, extractive_summary_text, abstractive_rouge, extractive_rouge
