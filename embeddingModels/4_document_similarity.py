from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents = [
    "virat is husband of anushka",
    "my girlfriend is happiness",
    "ashish girlfriend is avishi",
    "somi just studies"
]

query="what do you know about ashish"

doc_embedding=  embedding.embed_documents[documents]
query_embedding = embedding.embed_query[query]


scores = cosine_similarity([query_embedding],doc_embedding)[0]
print(scores)
index,result_score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(documents[index])