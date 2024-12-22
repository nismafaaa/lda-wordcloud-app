import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from google_play_scraper import reviews, Sort
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
import ast

# Step 1: Function to scrape reviews based on user input URL
def scrape_reviews(url):
    match = re.search(r'id=([^&]+)', url)
    if match:
        url_id = match.group(1)
        print("ID Aplikasi:", url_id)
    else:
        print("Tidak ditemukan ID aplikasi dalam URL.")
        return None

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
    df.to_csv('review.csv', encoding='utf-8')
    return df

# Step 2: Preprocessing function for cleaning text
def clean_text(text):
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

# Function to change abbreviations based on a dictionary
def change_text(text, abbreviation_dict):
    words = text.split()
    cleaned_words = [abbreviation_dict.get(word, word) for word in words]
    return ' '.join(cleaned_words)

# Step 3: Tokenizing, stemming, and stopword removal
nltk.download('punkt')

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(text):
    stop_factory = StopWordRemoverFactory().get_stop_words()
    more_stopword = ['kalau', 'untuk', 'duh', 'nih', 'menjadi', 'yang', 'kita', 'mas', 'guys', 'bapak', 'kamu', 'jadi', 'buat', 'are', 'it', 'is', 'of', 'nya', 'no', 'sebuah', 'at', 'that', 'but', 'sama', 'cuma', 'kadang', 'deh', 'wah']
    data = stop_factory + more_stopword
    filtered_text = ' '.join([word for word in text.split() if word not in data])
    return filtered_text

# Create a stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_text(text):
    return stemmer.stem(text)

# Step 4: Generate LDA model and word cloud
def generate_lda_and_wordcloud(df_token):
    # Pastikan tidak menggunakan ast.literal_eval jika data sudah berupa list Python
    # df_token['tokenized_stemmed_text'] = df_token['tokenized_stemmed_text'].apply(ast.literal_eval)

    # Create dictionary and corpus for LDA
    dictionary = corpora.Dictionary(df_token['tokenized_stemmed_text'])
    corpus = [dictionary.doc2bow(doc) for doc in df_token['tokenized_stemmed_text']]

    # Create bigram and trigram models
    bigram = Phrases(df_token['tokenized_stemmed_text'], min_count=5, threshold=100)
    trigram = Phrases(bigram[df_token['tokenized_stemmed_text']], threshold=100)

    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    # Preprocess: filter extremes, bigrams
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    df_token['tokenized_stemmed_text'] = df_token['tokenized_stemmed_text'].apply(lambda x: trigram_mod[bigram_mod[x]])

    # Re-create corpus with filtered dictionary
    corpus = [dictionary.doc2bow(doc) for doc in df_token['tokenized_stemmed_text']]

    # Train and evaluate multiple models
    best_coherence = 0
    best_model = None

    for num in range(2, 6):
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num, random_state=0, passes=20, alpha='auto')
        coherence_model = CoherenceModel(model=lda_model, texts=df_token['tokenized_stemmed_text'], dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        print(f"Num Topics: {num}, Coherence Score: {coherence_score}")

        if coherence_score > best_coherence:
            best_coherence = coherence_score
            best_model = lda_model

    print("Best Coherence Score:", best_coherence)

    # Function to generate word cloud for each topic
    def plot_word_cloud(lda_model, num_topics):
        for i in range(num_topics):
            plt.figure(figsize=(8, 6))
            topic_words = dict(lda_model.show_topic(i, topn=20))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Topic {i + 1}")
            plt.show()

    plot_word_cloud(best_model, best_model.num_topics)

    best_model.save("lda_model.model")
    dictionary.save("lda_dictionary.dict")
    with open("lda_corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)

# Step 5: Main function to integrate everything
def main():
    url = input("Masukkan URL aplikasi di Google Play Store: ")
    df = scrape_reviews(url)

    if df is not None:
        # Preprocess and clean reviews
        df_kamus = pd.read_csv('kamus_singkatan.csv', delimiter=';')
        abbreviation_dict = dict(zip(df_kamus['singkatan'], df_kamus['arti']))

        df_clean = df.copy()
        df_clean['content'] = df_clean['content'].apply(clean_text)
        df_clean['content'] = df_clean['content'].apply(lambda x: change_text(x, abbreviation_dict))
        df_unique = df_clean.drop_duplicates()

        df_token = df_unique.copy()
        df_token['tokenized_text'] = df_token['content'].apply(tokenize_text)
        df_token['stemmed_text'] = df_token['content'].apply(stem_text)
        df_token['filtered_text'] = df_token['stemmed_text'].apply(remove_stopwords)
        df_token['tokenized_stemmed_text'] = df_token['filtered_text'].apply(word_tokenize)

        df_token.to_csv('cleaned_review.csv', encoding='utf-8')

        # Generate LDA model and word cloud
        generate_lda_and_wordcloud(df_token)

if __name__ == "__main__":
    main()