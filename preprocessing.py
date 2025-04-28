import re
import os
import pandas as pd
import nltk
import time
import mlflow
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from config import logger, CLEANED_REVIEW_PATH

# Download required NLTK data
nltk.download('punkt', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self._initialize_stemmer()
        
    def _initialize_stemmer(self):
        """Initialize the Sastrawi stemmer."""
        try:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
            logger.info("Stemmer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize stemmer: {str(e)}")
            raise

    def clean_text(self, text):
        """Clean text by removing special characters and normalizing format"""
        text = re.sub(r'-', ' ', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'\w+\d+\w+', '', text)
        text = re.sub(r'[^a-zA-Z# \.]+', '', text)
        text = re.sub(r'\.|\.{2,}', '', text)
        text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
        text = text.lower()
        text = re.sub(r'&amp[.]?', 'dan', text)
        text = re.sub(r'\buser\b', '', text)
        return text.strip()

    def change_text(self, text, abbreviation_dict):
        """Replace abbreviations with their full forms"""
        words = text.split()
        cleaned_words = [abbreviation_dict.get(word, word) for word in words]
        return ' '.join(cleaned_words)

    def tokenize_text(self, text):
        """Tokenize text into individual words"""
        return word_tokenize(text)

    def stem_text(self, text):
        """Perform stemming using the initialized stemmer"""
        return self.stemmer.stem(text)

    def remove_stopwords(self, text):
        """Remove common stopwords from text"""
        stop_factory = StopWordRemoverFactory().get_stop_words()
        more_stopword = [
            'kalau', 'untuk', 'duh', 'nih', 'menjadi', 'yang', 'kita', 'mas', 
            'guys', 'bapak', 'kamu', 'jadi', 'buat', 'are', 'it', 'is', 'of', 
            'nya', 'no', 'sebuah', 'at', 'that', 'but', 'sama', 'cuma', 
            'kadang', 'deh', 'wah'
        ]
        stop_words = set(stop_factory + more_stopword)

        filtered_text = ' '.join([word for word in text.split() if word not in stop_words])
        return filtered_text
    
    def preprocess_reviews(self, df, abbreviation_dict):
        """Process review dataframe through all preprocessing steps"""
        try:
            start_time = time.time()
            df_clean = df.copy()

            # Apply cleaning functions
            df_clean['content'] = df_clean['content'].apply(self.clean_text)
            df_clean['content'] = df_clean['content'].apply(
                lambda x: self.change_text(x, abbreviation_dict))
            
            # Remove duplicates
            df_unique = df_clean.drop_duplicates()
            df_token = df_unique.copy()
            
            # Apply tokenization, stemming and stopword removal
            df_token['tokenized_text'] = df_token['content'].apply(self.tokenize_text)
            df_token['stemmed_text'] = df_token['content'].apply(self.stem_text)
            df_token['filtered_text'] = df_token['stemmed_text'].apply(self.remove_stopwords)
            df_token['tokenized_stemmed_text'] = df_token['filtered_text'].apply(self.tokenize_text)

            # Save processed data
            df_token.to_csv(CLEANED_REVIEW_PATH, encoding="utf-8", index=False)

            duration = time.time() - start_time
            mlflow.log_metric("preprocessing_duration_seconds", duration)

            return df_token

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise