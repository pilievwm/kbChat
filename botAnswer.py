import numpy as np
import openai
import pandas as pd
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


# Update the models to use
COMPLETIONS_MODEL = "gpt-3.5-turbo"
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
    
    return document_similarities[:5]

MAX_SECTION_LEN = 2900
MIN_SECTION_LEN = 20
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

f"Context separator contains {separator_len} tokens"

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

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
                
        chosen_sections.append(SEPARATOR + document_section.description.replace("\n", " ") + "\nLink: " + document_section.url.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
    
    header = """Act as CloudCart support agent. \n
                Let\'s think step by step trought the provided context. \n
                Provide the Link URL if it is shown at the answer! If the context is empty, say "No answer" \n
                \n
                Context:\n 
            """
    #header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A: """

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 1000,
    "model": COMPLETIONS_MODEL,
}

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
    if show_prompt:
        print("Context:", prompt)
    
    messages = [
    {"role": "system", "content": prompt},
    ]

    try:
        response = openai.ChatCompletion.create(
            messages=messages,
            **COMPLETIONS_API_PARAMS
        )
    except openai.error.RateLimitError:
        print("Please try again later.")
        return ""
    except TypeError:
        print("Please try again")
    except openai.error.InvalidRequestError as e:
        if "This model's maximum context length is 4097 tokens" in str(e):
            print("Please provide a shorter context.")
        else:
            print("Please try again later.")
        return ""

    return response["choices"][0]["message"]["content"]




def answer_bot(question):
    query = question
    answer = answer_query_with_context(query, df, document_embeddings)

    # your logic to generate the answer based on the question
    answer_bot = answer
    return answer_bot