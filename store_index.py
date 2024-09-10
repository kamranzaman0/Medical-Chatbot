from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
print(PINECONE_API_KEY)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
  name="m-chatbot",
  dimension=384,
  metric="cosine",
  spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )
)

#Creating Embeddings for Each of The Text Chunks & storing
index_name = "m-chatbot"

vectorstore = PineconeVectorStore.from_documents(
    text_chunks,
    index_name=index_name,
    embedding=embeddings
)
docsearch=PineconeVectorStore.from_existing_index(index_name, embeddings)
