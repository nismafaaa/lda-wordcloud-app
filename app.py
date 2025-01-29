from flask import Flask, request, render_template, redirect, url_for, flash
import logging
from test import ReviewLensApp  
import pandas as pd
import os

# Flask app initialization
app = Flask(__name__)
review_app = ReviewLensApp()

# Configures the logging to display the time, name, and level of log messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Web application routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        try:
            df = review_app.scrape_reviews(url)
            df_kamus = pd.read_csv('kamus_singkatan.csv', delimiter=';')
            abbreviation_dict = dict(zip(df_kamus['singkatan'], df_kamus['arti']))
            
            processed_df = review_app.preprocess_reviews(df, abbreviation_dict)
            model, dictionary, corpus = review_app.generate_lda_model(processed_df)
            
            return redirect(url_for('display_wordclouds'))
        
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            flash('An error occurred while processing your request. Please try again.', 'error')
            return redirect(url_for('index'))
    
    return render_template('index.html')

@app.route('/wordclouds')
def display_wordclouds():
    try:
        images = [
            f"wordclouds/{filename}" for filename in os.listdir("static/wordclouds")
            if filename.endswith(".png")
        ]
    except FileNotFoundError:
        logger.error("Wordclouds directory not found.")
        flash('No wordclouds found. Please generate one first.', 'info')
        return redirect(url_for('index'))
    
    return render_template('wordcloud.html', images=images)

# Application execution
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)