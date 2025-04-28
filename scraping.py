import time
import re
import pandas as pd
from google_play_scraper import reviews, Sort
import mlflow

from config import logger

def scrape_reviews(url):
    """Scrape reviews from Google Play Store URL with error handling"""
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

        duration = time.time() - start_time
        mlflow.log_metric("scrape_duration_seconds", duration)

        return df

    except Exception as e:
        mlflow.log_metric("scrape_errors_total", 1)
        logger.error(f"Error scraping reviews: {str(e)}")
        raise