import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from config import logger, ABBREVIATION_DICT_PATH, WORDCLOUD_OUTPUT_FOLDER

def load_abbreviation_dict():
    """Load abbreviation dictionary from CSV file"""
    try:
        df_kamus = pd.read_csv(ABBREVIATION_DICT_PATH, delimiter=';')
        abbreviation_dict = dict(zip(df_kamus['singkatan'], df_kamus['arti']))
        return abbreviation_dict
    except Exception as e:
        logger.error(f"Error loading abbreviation dictionary: {str(e)}")
        raise

def generate_wordclouds(lda_model, num_topics):
    """Generate and save wordclouds for each topic"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(WORDCLOUD_OUTPUT_FOLDER, exist_ok=True)

        for i in range(num_topics):
            topic_words = dict(lda_model.show_topic(i, topn=20))
            
            # Generate wordcloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
            
            # Save wordcloud image
            output_path = os.path.join(WORDCLOUD_OUTPUT_FOLDER, f"wordcloud_topic_{i + 1}.png")
            wordcloud.to_file(output_path)
            logger.info(f"Word cloud for Topic {i + 1} saved to {output_path}")
            
    except Exception as e:
        logger.error(f"Error generating wordclouds: {str(e)}")
        raise