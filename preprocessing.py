# #pip install sastrawi
# # pip install --user -U nltk
# import pandas as pd
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
# nltk.download('punkt_tab')

# df_kamus = pd.read_csv('kamus_singkatan.csv', delimiter=';')
# abbreviation_dict = dict(zip(df_kamus['singkatan'], df_kamus['arti']))

# def clean_text(text):
#   # Replace hyphen (-) with space
#   text = re.sub(r'-', ' ', text)

#   # Remove symbol hashtag but preserve the word inside hashtag
#   text = re.sub(r'#(\w+)', r'\1', text)

#   # Remove words containing numbers
#   text = re.sub(r'\w+\d+\w+', '', text)

#   # Remove remaining non-alphanumeric characters (excluding hashtags)
#   text = re.sub(r'[^a-zA-Z# \.]+', '', text)

#   # Remove all periods (single or multiple)
#   text = re.sub(r'\.|\.{2,}', '', text)

#   # Split camel-case words
#   text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)

#   # Convert text to lowercase
#   text = text.lower()

#   # Replace &amp with 'dan'
#   text = re.sub(r'&amp[.]?', 'dan', text)

#   # Remove word 'user'
#   text = re.sub(r'\buser\b', '', text)

#   return text.strip()

# def change_text(text, abbreviation_dict):
#     words = text.split()
#     cleaned_words = [abbreviation_dict.get(word, word) for word in words]
#     return ' '.join(cleaned_words)

# df = pd.read_csv('review.csv')

# df_clean = df.copy()
# df_clean['content'] = df_clean['content'].apply(clean_text)
# df_clean['content'] = df_clean['content'].apply(lambda x: change_text(x, abbreviation_dict))
# df_unique = df_clean.drop_duplicates()
# df_unique.head()

# def tokenize_text(text):
#   tokens = word_tokenize(text)
#   return tokens

# def remove_stopwords(text):
#   #load default stopword
#   stop_factory = StopWordRemoverFactory().get_stop_words()
#   more_stopword = ['kalau', 'untuk', 'duh', 'nih', 'menjadi', 'yang', 'kita',
#                    'mas', 'guys', 'bapak', 'kamu', 'jadi', 'buat', 'are', 'it',
#                    'is', 'of', 'nya', 'no', 'sebuah', 'at', 'that', 'but', 'sama'
#                    'cuma' 'kadang', 'deh', 'wah'
#                    ]

#   #menambahkan stopword
#   data = stop_factory + more_stopword
#   #menggabungkan stopword
#   filtered_text = ' '.join([word for word in text.split() if word not in data])
#   #tokens = nltk.word_tokenize(filtered_text)
#   return filtered_text

# # Create a stemmer
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

# # Define a function to perform stemming
# def stem_text(text):
#     return stemmer.stem(text)

# df_token = df_unique.copy()
# df_token['tokenized_text'] = df_token['content'].apply(tokenize_text)
# df_token['stemmed_text'] = df_token['content'].apply(stem_text)
# df_token['filtered_text'] = df_token['stemmed_text'].apply(remove_stopwords)
# df_token['tokenized_stemmed_text'] = df_token['filtered_text'].apply(word_tokenize)

# df_token.to_csv('cleaned_review.csv', encoding='utf-8')