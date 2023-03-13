import pandas as pd
import nltk
nltk.download('punkt')

df = pd.read_csv('data/articles_count.csv')
df['tokens'] = df['description'].apply(lambda x: len(nltk.word_tokenize(x)))
df.to_csv('data/articles_count.csv', index=False)