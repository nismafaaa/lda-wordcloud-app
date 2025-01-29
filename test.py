# Import the necessary Python library
import time
import pandas as pd
import re
import os
import nltk
import logging
from prometheus_client import start_http_server, Counter, Summary, Gauge
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from google_play_scraper import reviews, Sort
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim.models.phrases import Phrases, Phraser
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configure logging to help debugging and tracking the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialization prometheus metrics
SCRAPE_REQUESTS = Counter('scrape_requests_total', 'Total number of scrape requests')
SCRAPE_ERRORS = Counter('scrape_errors_total', 'Total number of scrape errors')
SCRAPE_DURATION = Summary('scrape_duration_seconds', 'Time taken to scrape reviews')
PREPROCESSING_DURATION = Summary('preprocessing_duration_seconds', 'Time taken for preprocessing')
LDA_DURATION = Summary('lda_duration_seconds', 'Time taken to generate LDA model')
REVIEW_COUNT = Gauge('reviews_processed_total', 'Number of reviews processed')
MODEL_COHERENCE = Gauge('model_coherence_score', 'Current model coherence score')

nltk.download('punkt')

class ReviewLensApp:
    #Initialize prometheus metrics server
    def __init__(self):
        self.metrics_port = int(os.getenv("APP_METRICS_PORT", 8000))

    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.metrics_port)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")
            raise
    
    #Scrapes reviews from a Google Play Store URL
    @SCRAPE_DURATION.time()  
    def scrape_reviews(self, url):
        """Scrape reviews with error handling"""
        try:
            start_time = time.time()
            match = re.search(r'id=([^&]+)', url)
            if not match:
                raise ValueError("Invalid Google Play Store URL")

            url_id = match.group(1)
            logger.info(f"Scraping reviews for app ID: {url_id}")

            result, _ = reviews(
                url_id,
                lang='id',
                country='id',
                sort=Sort.NEWEST,
                count=1000,
                filter_score_with=None
            )

            df = pd.DataFrame(result)
            df = df[['content']]

            REVIEW_COUNT.set(len(df))
            SCRAPE_DURATION.observe(time.time() - start_time)
            SCRAPE_REQUESTS.inc()

            return df

        except Exception as e:
            logger.error(f"Error scraping reviews: {str(e)}")
            raise
    
    # Cleans and preprocess the text data from reviews
    def clean_text(self, text):
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
        words = text.split()
        cleaned_words = [abbreviation_dict.get(word, word) for word in words]
        return ' '.join(cleaned_words)

    def tokenize_text(self, text):
        return word_tokenize(text)

    def _initialize_stemmer(self):
        """Initialize the Sastrawi stemmer."""
        try:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
            logger.info("Stemmer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize stemmer: {str(e)}")
            raise

    def remove_stopwords(self, text):
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

    def _initialize_stemmer(self):
        """Initialize the Sastrawi stemmer."""
        try:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
            logger.info("Stemmer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize stemmer: {str(e)}")
            raise
        
    def stem_text(self, text):
        """Perform stemming using the initialized stemmer."""
        if not hasattr(self, 'stemmer'):
            self._initialize_stemmer()  
        return self.stemmer.stem(text)

    @PREPROCESSING_DURATION.time()
    def preprocess_reviews(self, df, abbreviation_dict):
        """Enhanced preprocessing with better error handling"""
        try:
            start_time = time.time()
            df_clean = df.copy()

            df_clean['content'] = df_clean['content'].apply(self.clean_text)
            df_clean['content'] = df_clean['content'].apply(
                lambda x: self.change_text(x, abbreviation_dict))
            df_unique = df_clean.drop_duplicates()
            df_token = df_unique.copy()
            df_token['tokenized_text'] = df_token['content'].apply(self.tokenize_text)
            df_token['stemmed_text'] = df_token['content'].apply(self.stem_text)
            df_token['filtered_text'] = df_token['stemmed_text'].apply(self.remove_stopwords)
            df_token['tokenized_stemmed_text'] = df_token['filtered_text'].apply(self.tokenize_text)

            file_path = os.path.join(os.getcwd(), "cleaned_review.csv")
            df_token.to_csv(file_path, encoding="utf-8", index=False)

            PREPROCESSING_DURATION.observe(time.time() - start_time)

            return df_token

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    # Function to generates and saves wordclouds for each topic
    def save_and_plot_word_cloud(self, lda_model, num_topics):
        output_folder = os.path.join('static', 'wordclouds')
        os.makedirs(output_folder, exist_ok=True) 

        for i in range(num_topics):
            topic_words = dict(lda_model.show_topic(i, topn=20))
            
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
            
            output_path = os.path.join(output_folder, f"wordcloud_topic_{i + 1}.png")
            wordcloud.to_file(output_path)
            print(f"Word cloud for Topic {i + 1} saved to {output_path}")
            
            plt.figure(figsize=(8, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Topic {i + 1}")
            plt.show()

    # Building LDA model to discover topics withing the reviews
    @LDA_DURATION.time()
    def generate_lda_model(self, df_token):
        """Generate LDA model with improved metrics and logging"""
        try:
            start_time = time.time()
            bigram = Phrases(df_token['tokenized_stemmed_text'], min_count=5, threshold=100)
            trigram = Phrases(bigram[df_token['tokenized_stemmed_text']], threshold=100)

            bigram_mod = Phraser(bigram)
            trigram_mod = Phraser(trigram)

            dictionary = corpora.Dictionary(df_token['tokenized_stemmed_text'])
            dictionary.filter_extremes(no_below=5, no_above=0.5)

            df_token['tokenized_stemmed_text'] = df_token['tokenized_stemmed_text'].apply(
                lambda x: trigram_mod[bigram_mod[x]]
            )

            corpus = [dictionary.doc2bow(doc) for doc in df_token['tokenized_stemmed_text']]

            best_coherence = 0
            best_model = None
            best_num_topics = 0

            for num_topics in range(2, 6):
                model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    random_state=0,
                    passes=20,
                    alpha='auto'
                )

                coherence_model = CoherenceModel(
                    model=model,
                    texts=df_token['tokenized_stemmed_text'],
                    dictionary=dictionary,
                    coherence='c_v'
                )

                coherence_score = coherence_model.get_coherence()

                if coherence_score > best_coherence:
                    best_coherence = coherence_score
                    best_model = model
                    best_num_topics = num_topics

            if best_model:
                self.save_and_plot_word_cloud(best_model, best_num_topics)
            
            LDA_DURATION.observe(time.time() - start_time)

            return best_model, dictionary, corpus

        except Exception as e:
            logger.error(f"Error generating LDA model: {str(e)}")
            raise
    
    # Main application loop
    # Continuously prompts the user to enter  a Google Play Store URL, processes the reviews, and performs LDA analysis.
    def run(self):
        self.start_metrics_server()
        """Main application loop with improved error handling"""
        while True:
            try:
                url = input("Enter Google Play Store URL (or 'quit' to exit): ")
                if url.lower() == 'quit':
                    break

                df = self.scrape_reviews(url)
                df_kamus = pd.read_csv('kamus_singkatan.csv', delimiter=';')
                abbreviation_dict = dict(zip(df_kamus['singkatan'], df_kamus['arti']))

                processed_df = self.preprocess_reviews(df, abbreviation_dict)
                model, dictionary, corpus = self.generate_lda_model(processed_df)

            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                continue

# Scripts execution
if __name__ == "__main__":
    app = ReviewLensApp()
    app.run()