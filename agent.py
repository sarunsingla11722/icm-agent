import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DB_DIR = "./chroma_db"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def handle_query():
    print("Loading local vector database...")

    if not os.path.exists(CHROMA_DB_DIR):
        print(f"ERROR: No vector database found at {CHROMA_DB_DIR}")
        print("Please run ingest.py first to build the knowledge base.")
        return

    if "GOOGLE_API_KEY" not in os.environ:
         print("ERROR: GOOGLE_API_KEY environment variable is not set.")
         print("Create a .env file and add: GOOGLE_API_KEY=your_gemini_api_key")
         return

    # Use the same embedding model we used to ingest the documents
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

    # Create the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Get top 5 most relevant chunks

    # Initialize the LLM (Gemini 2.5 Pro)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

    # Define the prompt
    template = """You are a helpful assistant that answers questions based on the user's private Google Drive documents.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer based on the context, just say that you don't know.
Keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Create the modern LCEL RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n========================================================")
    print("Drive QA Agent Ready! Type your question below.")
    print("Type 'exit' or 'quit' to stop.")
    print("========================================================\n")

    while True:
        try:
            query = input("Ask a question: ")
            if query.lower() in ("exit", "quit", "q"):
                 print("Goodbye!")
                 break
            if not query.strip():
                 continue

            print("\nThinking...")
            # We fetch retrieved docs separately first, just to print citations
            retrieved_docs = retriever.invoke(query)

            # Now we generate the answer
            answer = rag_chain.invoke(query)

            print("\n🤖 ANSWER:")
            print(answer)

            print("\n📚 SOURCES:")
            for doc in retrieved_docs:
                 source = doc.metadata.get("title", "Unknown Document") # drive loader usually puts title
                 print(f"- {source}")
            print("\n" + "-"*40 + "\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")

if __name__ == "__main__":
    handle_query()
