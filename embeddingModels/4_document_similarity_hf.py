from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Choose a sentence-transformer model (must support feature extraction)
client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token=HUGGINGFACEHUB_API_TOKEN
)

documents = [
    "virat is husband of anushka",
    "my girlfriend is happiness",
    "ashish girlfriend is avishi",
    "somi just studies"
]

query = "what do you know about ashish"

# Get embeddings
doc_embeddings = [client.feature_extraction(text) for text in documents]
query_embedding = client.feature_extraction(query)

# Calculate cosine similarities
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find best match
index, result_score = max(enumerate(scores), key=lambda x: x[1])

print("Similarity Scores:", scores)
print("Most similar doc:", documents[index])
