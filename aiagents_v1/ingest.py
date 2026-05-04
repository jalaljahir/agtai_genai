from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# load markdoen files
loader = DirectoryLoader('./knowledge_base', glob='**/*.md')
documents = loader.load()

# split documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

#use ollama for embedding : use Mistral by default
embeddings = OllamaEmbeddings(model="mistral:7b-instruct-v0.3-q4_K_M")

# create vector store to store the embeddings
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory='./chroma_db',
    collection_name="knowledge_base")
print("Knowledge base indexed! Ingestion completed successfully!")