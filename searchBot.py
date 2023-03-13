import numpy as np
import openai
import pandas as pd
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


# Update the models to use
EMBEDDING_MODEL = "text-embedding-ada-002"

# Read the CSV into a pandas DataFrame
data_dir = '/app/data'
df = pd.read_csv(os.path.join(data_dir, 'articles_count.csv'))
df = df.set_index(["title"])

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    try:
        result = openai.Embedding.create(
          model=model,
          input=text
        )
        return result["data"][0]["embedding"]
    except openai.error.RateLimitError:
        print("Please try again later.")

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title"])
    return {
           (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

# Compute the embeddings for each row in the DataFrame
document_embeddings = load_embeddings(os.path.join(data_dir, 'embeddings.csv'))

def vector_similarity(x: list[float], y: list[float]) -> float:
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities[:1]

MAX_SECTION_LEN = 1900
MIN_SECTION_LEN = 20
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"

def construct_prompt(search: str, context_embeddings: dict, df: pd.DataFrame) -> str:

    most_relevant_document_sections = order_document_sections_by_query_similarity(search, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[section_index]
        section_length = len(document_section.description.split())
        if section_length < MIN_SECTION_LEN:
            continue
        chosen_sections_len += section_length + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
                
        chosen_sections.append(document_section.description.replace("\n", " ") + "\nLink: " + document_section.url.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
    
    return "\n".join(chosen_sections)

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:

    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    return prompt



def answer_bot(search):
    query = search
    answer = answer_query_with_context(query, df, document_embeddings)

    # your logic to generate the answer based on the question
    answer_bot = answer
    return answer_bot