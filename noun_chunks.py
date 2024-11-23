import pandas as pd
import spacy

# Load the uploaded CSV file
file_path = 'news_id_to_content.csv'
data = pd.read_csv(file_path)

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Function to extract noun chunks or noun words representing the sentence
def extract_noun_chunks(text):
    """
    Extract noun chunks or representative nouns from a given text.
    Args:
        text (str): The input sentence.
    Returns:
        list: A list of noun chunks or nouns representing the sentence.
    """
    doc = nlp(text)
    # Extract noun chunks and nouns
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    return noun_chunks

# Process the 'Content' column to extract noun chunks
if 'Content' in data.columns:
    data['Noun Chunks'] = data['Content'].apply(lambda x: extract_noun_chunks(str(x)))

# Save the updated DataFrame with noun chunks to a CSV file
output_path = 'news_with_noun_chunks.csv'
data.to_csv(output_path, index=False)

output_path