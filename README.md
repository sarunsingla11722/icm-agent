'# Google Drive QA Agent

This is a custom Retrieval-Augmented Generation (RAG) agent that allows you to chat with your Google Drive documents locally.

Built using Python, LangChain, ChromaDB, and Google Gemini Models.

## Prerequisites

You need three things to run this:
1. Python 3.9+ with dependencies installed. You can install them by running the following in your terminal from this directory:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. A Gemini API Key (`GOOGLE_API_KEY`).
3. Google Cloud OAuth 2.0 Credentials for reading Drive (`credentials.json`).

### 1. Gemini API Key
* Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
* Generate an API key.
* Create a `.env` file in this directory and add:
  ```
  GOOGLE_API_KEY=your_copied_api_key
  ```

### 2. Google Drive API Credentials (credentials.json)
In order for the script to access your Drive files, you need to authorize it via GCP.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new Project or select an existing one.
3. Search for **"Google Drive API"** and click **Enable**.
4. Go to **APIs & Services > Credentials**.
5. Click **Create Credentials > OAuth client ID**.
6. Set the Application type to **Desktop app**. Name it "Drive Agent".
7. Click **Create**, then click **Download JSON**.
8. Save that downloaded file as `credentials.json` directly inside this project folder (`~/.gemini/jetski/scratch/drive-qa-agent/credentials.json`).

> **Note on OAuth Consent Screen:** You may be asked to configure an OAuth consent screen on GCP. Choose "External", add yourself to "Test Users", and add the scope `https://www.googleapis.com/auth/drive.readonly`.

## Usage

### Step 1: Ingest Documents into the Vector Database

First, find the ID of the Google Drive folder you want to ingest. You can find this by opening the folder in your web browser and looking at the URL:
`https://drive.google.com/drive/u/0/folders/YOUR_FOLDER_ID_HERE`

Run the ingest script:
```bash
python ingest.py YOUR_FOLDER_ID_HERE
```
_The first time you run this, a browser window will open asking you to log in with your Google account. This will generate a local `token.json` file so you don't have to log in every time._

### Step 2: Chat with Your Documents

Once the ChromaDB vector store is populated, launch the agent:
```bash
python agent.py
```
Type your questions and the agent will respond using Gemini 1.5 Pro and your retrieved documents.
