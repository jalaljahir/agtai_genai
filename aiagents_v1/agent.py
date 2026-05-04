# from langchain.chains import RetrievalQA

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# load the vector store
embedding = OllamaEmbeddings(model="mistral:7b-instruct-v0.3-q4_K_M")
db = Chroma(
    persist_directory='./chroma_db',
    embedding_function=embedding,
    collection_name="knowledge_base")

# create retriever from the vector store
retriever = db.as_retriever()

# create the LLM
llm = OllamaLLM(model="mistral:7b-instruct-v0.3-q4_K_M")

# # create the RetrievalQA chain
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True)

# define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """You are an AI assistant that answers questions based on the provided context. 
    Use the retrieved documents to answer the question as accurately as possible. 
    If you don't know the answer, say you don't know. 
    Always provide the source of your information from the retrieved documents.

    Question: {question}
    Context: {context}
    """)

#build the chain manually to have more control over the prompt and output parsing
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
    # | (lambda x: {"result": x, "source_documents": retriever.get_relevant_documents(x)})
)

# Run interactive loop to ask questions
print("Ask questions about the knowledge base. Type 'exit' to quit.")

while True:
    query = input("You: ")
    if query.lower() in  ['quit', 'exit']:
        print("Exiting. Goodbye!")
        break
    # Pass 'query' directly as a string
    try:

        result = qa_chain.invoke(query)
        
        #safely get the answer and source documents from the result
        if isinstance(result, dict):
            answer = result.get('result', 'No answer found.')
            source_docs = result.get('source_documents', [])
        else:
            answer = str(result)
            source_docs = []
        
        print(f"AI: {answer}")
        if source_docs:
            print("\nSources:", [doc.metadata.get('source') for doc in source_docs])
        # else:
        #     print("\nNo source documents found.")
    except Exception as e:
        print(f"An error occurred: {e}")