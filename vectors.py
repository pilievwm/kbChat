import numpy as np
import openai
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002"

df = pd.read_csv('data/all_arcticles.csv', keep_default_na=False).replace('\n', ' ', regex=True)
df['tokens'] = df['description'].apply(lambda x: len(x.split()))

# Create a new DataFrame that contains only the description and token_count columns
count_df = df[['title', 'description', 'url', 'tokens']].copy()

# Save the new DataFrame to a CSV file
count_df.to_csv('data/articles_count.csv', index=False)

# Continue with the rest of your code using the new CSV file
count_df = count_df.set_index(["title"])

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    return {
        idx: get_embedding(r.description) for idx, r in df.iterrows()
    }

def save_embeddings_to_csv(embeddings: dict, fname: str):
    num_cols = len(list(embeddings.values())[0])
    columns = [str(i) for i in range(num_cols)]
    count_df = pd.DataFrame.from_dict(embeddings, orient='index', columns=columns)
    count_df.index.name = "title"  # Add this line to set the index name
    count_df.reset_index(inplace=True)
    count_df.to_csv(fname, index=False)

embeddings = compute_doc_embeddings(count_df)
save_embeddings_to_csv(embeddings, "data/embeddings.csv")

