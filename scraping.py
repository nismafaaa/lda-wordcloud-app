#pip install google-play-scraper
from google_play_scraper import app, Sort, reviews
import pandas as pd
import re

# URL string
url = "https://play.google.com/store/apps/details?id=com.tomoro.indonesia.android&hl=id&pli=1"

# Ekstraksi teks antara 'id=' dan '&'
match = re.search(r'id=([^&]+)', url)

if match:
    url_id = match.group(1)
    print("ID Aplikasi:", url_id)
else:
    print("Tidak ditemukan ID aplikasi dalam URL.")

result, countinuation_token = reviews(
    url_id,
    lang='id',
    country='id',
    sort = Sort.NEWEST,
    count = 1000,
    filter_score_with = None
    )

df = pd.DataFrame(result)

df = df[['content']]

df.to_csv('review.csv', encoding='utf-8')