# pip install gensim
# pip install wordcloud

import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
import ast

df_token = pd.read_csv('cleaned_review.csv')

# Create dictionary and corpus for LDA using the tokenized stemmed text
df_token['tokenized_stemmed_text'] = df_token['tokenized_stemmed_text'].apply(ast.literal_eval)
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

for num in range(2, 10):
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

        # Extract words and their corresponding weights for the topic
        topic_words = dict(lda_model.show_topic(i, topn=20))

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)

        # Display the word cloud
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Topic {i + 1}")
        plt.show()

# Use the best model to generate word clouds for each topic
plot_word_cloud(best_model, best_model.num_topics)

best_model.save("lda_model.model")
print("LDA model saved successfully.")

# Save the dictionary
dictionary.save("lda_dictionary.dict")
print("Dictionary saved successfully.")

# Save the corpus
with open("lda_corpus.pkl", "wb") as f:
    pickle.dump(corpus, f)
print("Corpus saved successfully.")