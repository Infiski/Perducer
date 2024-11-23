import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import util

# Load the news content file
news_content_file_path = 'news_id_to_content.csv'
news_content_data = pd.read_csv(news_content_file_path)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to preprocess text
def preprocess_text(text):
    """
    Preprocess the text by splitting it into words, removing stop words, and tokenizing.
    """
    tokens = text.split()
    tokens = [word for word in tokens if word.lower() not in ENGLISH_STOP_WORDS]
    return tokens

# Function to compute BERT embeddings
def get_bert_embedding(text):
    """
    Get BERT embedding for a given text.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the CLS token embedding as the sentence embedding
    return outputs.last_hidden_state[:, 0, :]

# Function to get top keywords based on BERT similarity
def extract_top_keywords(sentence, n=5):
    """
    Extract top n keywords from the given sentence that are most similar to the BERT embedding of the sentence.
    """
    sentence_embedding = get_bert_embedding(sentence)
    words = preprocess_text(sentence)
    
    if len(words) < n:
        n = len(words)  # Adjust n if fewer words are available
    
    if n == 0:
        return []  # Return empty list if no valid words are found
    
    word_embeddings = [get_bert_embedding(word) for word in words]
    word_embeddings = torch.cat(word_embeddings, dim=0)
    
    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(sentence_embedding, word_embeddings).squeeze(0)
    
    # Get top n words based on similarity
    top_indices = similarities.topk(n).indices
    top_keywords = [words[idx] for idx in top_indices]
    return top_keywords

# Prepare top 5 keywords for each news article
news_keywords = []
for _, row in news_content_data.iterrows():
    news_id = row['News ID']
    content = row['Content']
    
    # Extract top 5 keywords
    top_keywords = extract_top_keywords(content, n=5)
    news_keywords.append({'News ID': news_id, 'Top Keywords': ','.join(top_keywords) if top_keywords else 'None'})

# Create a DataFrame with results
news_keywords_df = pd.DataFrame(news_keywords)

# Save to a CSV file
output_path = 'news_keywords.csv'
news_keywords_df.to_csv(output_path, index=False)

print(f"Labeled news keywords saved to: {output_path}")
