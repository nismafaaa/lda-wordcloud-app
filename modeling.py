import time
import os
import mlflow
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim.models.phrases import Phrases, Phraser

from config import logger, LDA_NUM_TOPICS_RANGE, LDA_PASSES, WORDCLOUD_OUTPUT_FOLDER
from utils import generate_wordclouds

def generate_lda_model(df_token):
    """Generate LDA model and find optimal number of topics"""
    try:
        start_time = time.time()

        # Create bigrams and trigrams
        bigram = Phrases(df_token['tokenized_stemmed_text'], min_count=5, threshold=100)
        trigram = Phrases(bigram[df_token['tokenized_stemmed_text']], threshold=100)

        bigram_mod = Phraser(bigram)
        trigram_mod = Phraser(trigram)

        # Create dictionary
        dictionary = corpora.Dictionary(df_token['tokenized_stemmed_text'])
        dictionary.filter_extremes(no_below=5, no_above=0.5)

        # Apply trigram transformation
        df_token['tokenized_stemmed_text'] = df_token['tokenized_stemmed_text'].apply(
            lambda x: trigram_mod[bigram_mod[x]]
        )

        # Create corpus
        corpus = [dictionary.doc2bow(doc) for doc in df_token['tokenized_stemmed_text']]

        # Find optimal number of topics
        best_coherence = 0
        best_model = None
        best_num_topics = 0

        for num_topics in LDA_NUM_TOPICS_RANGE:
            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=0,
                passes=LDA_PASSES,
                alpha='auto'
            )

            coherence_model = CoherenceModel(
                model=model,
                texts=df_token['tokenized_stemmed_text'],
                dictionary=dictionary,
                coherence='c_v'
            )

            coherence_score = coherence_model.get_coherence()
            
            logger.info(f"Topics: {num_topics}, Coherence Score: {coherence_score}")

            if coherence_score > best_coherence:
                best_coherence = coherence_score
                best_model = model
                best_num_topics = num_topics

        # Log parameters and metrics in MLflow
        mlflow.log_param("lda_num_topics_range", list(LDA_NUM_TOPICS_RANGE))
        mlflow.log_param("lda_passes", LDA_PASSES)
        mlflow.log_param("best_num_topics", best_num_topics)
        mlflow.log_metric("model_coherence_score", best_coherence)

        duration = time.time() - start_time
        mlflow.log_metric("lda_duration_seconds", duration)

        # Generate word clouds for the best model
        if best_model:
            generate_wordclouds(best_model, best_num_topics)
            for i in range(best_num_topics):
                wordcloud_path = os.path.join(WORDCLOUD_OUTPUT_FOLDER, f"wordcloud_topic_{i + 1}.png")
                mlflow.log_artifact(wordcloud_path)  # Log word cloud as artifact

        return best_model, dictionary, corpus

    except Exception as e:
        logger.error(f"Error generating LDA model: {str(e)}")
        raise