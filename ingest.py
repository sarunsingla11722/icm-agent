import os
from dotenv import load_dotenv
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# We will look for a google_credentials.json file in the current directory
CREDENTIALS_FILE = "credentials.json"
# The token.json file will be automatically generated upon first successful auth
TOKEN_FILE = "token.json"

CHROMA_DB_DIR = "./chroma_db"

def ingest_drive_folder(folder_id):
    """
    Authenticates with Google Drive and ingest all files inside the specified folder.
    """
    print(f"Loading documents from Google Drive Folder ID: {folder_id}")

    if not os.path.exists(CREDENTIALS_FILE):
        print(f"ERROR: {CREDENTIALS_FILE} not found in the current directory.")
        print("Please obtain OAuth 2.0 Client credentials from the Google Cloud Console,")
        print("download the JSON file, and save it as credentials.json here.")
        return

    # Workaround for LangChain GoogleDriveLoader bug with user credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_FILE

    # Initialize the Google Drive Loader
    # It will automatically handle the OAuth flow in the browser if token.json is missing
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        recursive=True,  # Set to True so it checks subfolders
        credentials_path=CREDENTIALS_FILE,
        token_path=TOKEN_FILE,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

    print("Fetching documents...")
    try:
        docs = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    print(f"Loaded {len(docs)} documents.")
    if len(docs) == 0:
        print("No documents found to process. Exiting.")
        return

    print("Splitting documents into chunks...")
    # Split text into manageable chunks so the LLM doesn't overflow context windows
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    print("Generating embeddings and saving to ChromaDB...")
    # Using Gemini's embedding model
    # Note: Requires GOOGLE_API_KEY environment variable to be set

    if "GOOGLE_API_KEY" not in os.environ:
         print("ERROR: GOOGLE_API_KEY environment variable is not set.")
         print("Create a .env file and add: GOOGLE_API_KEY=your_gemini_api_key")
         return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print(f"Success! Embedded documents saved to {CHROMA_DB_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Google Drive folder into ChromaDB")
    parser.add_argument("folder_id", help="The ID of the Google Drive folder to ingest (found in the folder URL)")
    args = parser.parse_args()

    ingest_drive_folder(args.folder_id)
