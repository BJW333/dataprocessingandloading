import os
import re
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import spacy
import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
import argparse

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

#define stop words
stop_words = set(nlp.Defaults.stop_words)

#load BERT and GPT-2 models
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')  #set padding_side to 'left'
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
mainpathdir = Path('/Users/blakeweiss/Desktop/dataprocessingandloading')

# Set up logging
log_path = mainpathdir / 'processing.log'
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def extract_features(text):
    #Extract features from the text using BERT.
    inputs = bert_tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def generate_responses(prompts, max_length=50):
    #Generate responses to prompts using GPT2
    inputs = gpt2_tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    outputs = gpt2_model.generate(**inputs, max_length=max_length, pad_token_id=gpt2_tokenizer.eos_token_id)
    return [gpt2_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def is_valid(row):
    question, answer = row['question'], row['answer']
    if not isinstance(question, str) or not isinstance(answer, str):
        return False
    if len(question.split()) < 2 or len(answer.split()) < 2:
        return False

    question_features = extract_features(question).flatten()
    answer_features = extract_features(answer).flatten()
    similarity = cosine_similarity([question_features], [answer_features])[0][0]
    if similarity < 0.3:
        return False

    question_sentiment = TextBlob(question).sentiment.polarity
    answer_sentiment = TextBlob(answer).sentiment.polarity
    if question_sentiment * answer_sentiment < -0.5:
        return False

    return True

def enhance_conversation(prompts, batch_size=100):
    valid_data = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        enhanced_texts = generate_responses(batch_prompts, max_length=100)
        for prompt, enhanced_text in zip(batch_prompts, enhanced_texts):
            try:
                new_question, new_answer = enhanced_text.split("Answer:", 1)
                new_question = new_question.replace("Question:", "").strip()
                new_answer = new_answer.strip()
                if len(new_question.split()) >= 2 and len(new_answer.split()) >= 2:
                    valid_data.append({'question': new_question, 'answer': new_answer})
            except ValueError:
                continue
    return pd.DataFrame(valid_data)

def process_data(file_path):
    ext = Path(file_path).suffix
    if ext == '.csv':
        data = pd.read_csv(file_path)
    elif ext in ['.xls', '.xlsx']:
        data = pd.read_excel(file_path)
    elif ext == '.json':
        data = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")

    for col in data.columns:
        if data[col].dtype == 'object':
            data['processed_' + col] = data[col].apply(clean_text)

    if 'question' in data.columns and 'answer' in data.columns:
        data['valid'] = data.apply(is_valid, axis=1)
        valid_data = data[data['valid']]
        invalid_data = data[~data['valid']]
        logging.info(f"Invalid data discarded: {len(invalid_data)} rows")

        prompts = [
            f"Here is a question and its corresponding answer. Please refine the response to be more coherent and logical. \n\nQuestion: {row['question']} \nAnswer: {row['answer']} \nPlease improve coherence and completeness, and ensure the conversation makes sense."
            for _, row in valid_data.iterrows()
        ]

        enhanced_data = enhance_conversation(prompts, batch_size=200)
        
        if not enhanced_data.empty:
            for col in ['question', 'answer']:
                enhanced_data[f'polarity_{col}'] = enhanced_data[col].apply(lambda x: TextBlob(x).sentiment.polarity)
                enhanced_data[f'subjectivity_{col}'] = enhanced_data[col].apply(lambda x: TextBlob(x).sentiment.subjectivity)
                enhanced_data[f'word_count_{col}'] = enhanced_data[col].apply(lambda x: len(x.split()))
                enhanced_data[f'char_count_{col}'] = enhanced_data[col].apply(lambda x: len(x))
                enhanced_data[f'noun_chunks_{col}'] = enhanced_data[col].apply(lambda x: len(list(nlp(x).noun_chunks)))
                enhanced_data[f'features_{col}'] = enhanced_data[col].apply(lambda x: extract_features(x).tolist())

            #save only valid enhanced data
            inputs_file_path = mainpathdir / 'inputs.csv'
            outputs_file_path = mainpathdir / 'outputs.csv'
            enhanced_data.to_csv(inputs_file_path, index=False)
            enhanced_data.to_csv(outputs_file_path, index=False)

            logging.info(f"Processed inputs and outputs saved to {inputs_file_path} and {outputs_file_path}")

            return enhanced_data
        else:
            logging.error("No valid enhanced data found")
            return pd.DataFrame()
    else:
        logging.error("Data does not contain 'question' and 'answer' columns")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Text Data Processing Script")
    parser.add_argument('file_path', type=str, help='Path to the input data file')
    args = parser.parse_args()

    processed_data = process_data(args.file_path)
    if not processed_data.empty:
        print("Processed Data:\n", processed_data.head())

if __name__ == "__main__":
    main()
