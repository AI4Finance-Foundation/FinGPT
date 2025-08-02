from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline


# Load a local text-generation pipeline (make sure the model is downloaded)
local_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",  # You can use any compatible local model
    max_length=512
)

loader = PyPDFLoader("C:\\Users\\rahul\\OneDrive\\Desktop\\FinGPT-M\\Datasets\\Financial Documents\\2024_Annual_Report_Chevron.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=128, chunk_overlap=32)
docs = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(docs)}")
docs = docs[:70]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

llm = HuggingFacePipeline(pipeline=local_pipe)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Example queries for financial reports
queries = [
    "What was the net income for the year?",
    "Summarize the company's performance in Q4.",
    "List the main factors affecting revenue in 2024.",
    "What are the total assets reported?",
    "How much did operating expenses increase compared to last year?",
    "What is the company's earnings per share?",
    "Describe the company's cash flow situation.",
    "What are the key risks mentioned in the report?",
    "Who are the major shareholders?",
    "What is the outlook for the next fiscal year?",
    "How did the company's debt change in 2024?",
    "What is the dividend declared for shareholders?",
    "What new projects or investments are planned?",
    "How did international sales perform?",
    "What is the CEO's message to investors?"
]

# Example usage:
for q in queries:
    response = qa_chain.run(q)
    print(f"Q: {q}\nA: {response}\n")