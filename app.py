from flask import Flask, request, render_template, redirect, url_for, flash, session
import os
import json

from config import logger, WORDCLOUD_OUTPUT_FOLDER
from scraping import scrape_reviews
from preprocessing import TextPreprocessor
from modeling import generate_lda_model
from utils import load_abbreviation_dict

# Flask app initialization
app = Flask(__name__)
app.secret_key = os.urandom(24)  # for flash messages and session

# Initialize text preprocessor
text_preprocessor = TextPreprocessor()

# Web application routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        try:
            # Step 1: Scrape reviews
            df = scrape_reviews(url)
            logger.info(f"Successfully scraped {len(df)} reviews")
            
            # Step 2: Load abbreviation dictionary
            abbreviation_dict = load_abbreviation_dict()
            
            # Step 3: Preprocess reviews
            processed_df = text_preprocessor.preprocess_reviews(df, abbreviation_dict)
            logger.info(f"Successfully preprocessed {len(processed_df)} reviews")
            
            # Step 4: Generate LDA model
            model, dictionary, corpus = generate_lda_model(processed_df)
            logger.info("LDA model generation complete")
            
            # Store topic information in session for display
            topics_info = []
            for i in range(model.num_topics):
                topic_words = model.show_topic(i, topn=10)
                words = [word for word, _ in topic_words]
                topics_info.append({
                    "id": i + 1,
                    "words": ", ".join(words)
                })
            
            session['topics_info'] = topics_info
            
            return redirect(url_for('display_wordclouds'))
        
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            flash('An error occurred while processing your request. Please try again.', 'error')
            return redirect(url_for('index'))
    
    return render_template('index.html')

@app.route('/wordclouds')
def display_wordclouds():
    try:
        # Get list of wordcloud images
        wordcloud_dir = WORDCLOUD_OUTPUT_FOLDER.replace('static/', '')
        images = [
            f"{wordcloud_dir}/{filename}" for filename in os.listdir(WORDCLOUD_OUTPUT_FOLDER)
            if filename.endswith(".png")
        ]
        
        # Get topics information from session
        topics_info = session.get('topics_info', [])
        
    except FileNotFoundError:
        logger.error("Wordclouds directory not found.")
        flash('No wordclouds found. Please generate one first.', 'info')
        return redirect(url_for('index'))
    
    return render_template('wordcloud.html', images=images, topics=topics_info)

# Application execution
if __name__ == '__main__':
    # Make sure the wordclouds directory exists
    os.makedirs(WORDCLOUD_OUTPUT_FOLDER, exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=8000, debug=True)  # Changed port to 8000