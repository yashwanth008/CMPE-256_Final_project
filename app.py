# import os
# import json
# import faiss
# import numpy as np
# import requests
# import io
# import glob
# from fastapi import FastAPI, HTTPException
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# from pypdf import PdfReader
# import google.generativeai as genai

# # --- CONFIGURATION ---
# GOOGLE_API_KEY = "AIzaSyCtoV2QBnu3iai9XIWrCR58tSiIgpxnK-E"  # 🔴 PASTE KEY HERE
# INDEX_PATH = "index/papers.faiss"
# METADATA_PATH = "index/metadata.json"
# # --- CONFIGURATION ---
# PDF_CACHE_DIR = "pdfs"
# os.makedirs(PDF_CACHE_DIR, exist_ok=True)
# # Configure Google Gemini
# genai.configure(api_key=GOOGLE_API_KEY)
# # We use 1.5 Flash because it has a huge context window (can read whole books) and is fast/cheap
# model = genai.GenerativeModel('gemini-2.5-flash-lite')

# app = FastAPI()

# # Enable CORS for local testing
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- GLOBAL VARIABLES (Loaded on Startup) ---
# search_index = None
# search_model = None
# paper_metadata = []

# @app.on_event("startup")
# def load_resources():
#     global search_index, search_model, paper_metadata
#     print("⏳ Loading AI Models and FAISS Index...")
    
#     # 1. Load Embedding Model
#     search_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
#     # 2. Load FAISS Index
#     search_index = faiss.read_index(INDEX_PATH)
    
#     # 3. Load Metadata
#     with open(METADATA_PATH, "r", encoding="utf-8") as f:
#         paper_metadata = json.load(f)
        
#     print("✅ System Ready! Index contains", search_index.ntotal, "papers.")

# # --- HELPER FUNCTIONS (The "Agents") ---

# def get_pdf_text(url):
#     """
#     STRICT LOCAL MODE:
#     1. Extracts the ArXiv ID from the URL.
#     2. Searches the 'pdfs/' folder for ANY file matching that ID.
#     3. Reads the local file.
#     4. DOES NOT connect to the internet (No blocking errors).
#     """
#     try:
#         # 1. Extract the ID from the URL
#         # URL examples: 
#         # https://arxiv.org/pdf/2107.10390.pdf  -> ID: 2107.10390
#         # https://arxiv.org/abs/2107.10390      -> ID: 2107.10390
        
#         # Remove http/https and domain
#         clean_name = url.split("/")[-1] 
#         # Remove .pdf extension for matching
#         paper_id = clean_name.replace(".pdf", "")
#         # Remove version numbers if present (e.g., v1, v2) to find the main file
#         paper_id_core = paper_id.split('v')[0] 

#         print(f"🔍 Looking for local PDF matching ID: {paper_id_core}...")

#         # 2. Smart Search in 'pdfs' folder
#         # This looks for ANY file that starts with the ID
#         # So '2107.10390' will match '2107.10390.pdf', '2107.10390v1.pdf', etc.
#         search_pattern = os.path.join(PDF_CACHE_DIR, f"{paper_id_core}*.pdf")
#         found_files = glob.glob(search_pattern)

#         if not found_files:
#             print(f"❌ File not found in {PDF_CACHE_DIR}. Please download it manually.")
#             # OPTIONAL: Raise an error so the UI tells you to download it
#             raise FileNotFoundError(f"Please download {url} and save it to 'pdfs/' folder.")

#         # 3. Read the first matching file found
#         local_path = found_files[0]
#         print(f"📂 Found local file: {local_path}")
        
#         reader = PdfReader(local_path)
#         text = ""
#         # Read first 15 pages (increased slightly)
#         max_pages = min(len(reader.pages), 15)
#         for i in range(max_pages):
#             text += reader.pages[i].extract_text() + "\n"
            
#         return text

#     except Exception as e:
#         print(f"❌ Error processing local PDF: {e}")
#         return None

# # --- API ENDPOINTS ---

# class SearchQuery(BaseModel):
#     query: str

# class SummarizeRequest(BaseModel):
#     pdf_url: str
#     title: str

# @app.post("/search")
# def search_papers(payload: SearchQuery):
#     """Searches the FAISS index."""
#     query_vector = search_model.encode([payload.query], normalize_embeddings=True).astype('float32')
#     distances, indices = search_index.search(query_vector, k=5)
    
#     results = []
#     for i, idx in enumerate(indices[0]):
#         if idx == -1: continue
#         item = paper_metadata[idx]
        
#         # Clean ID for PDF link
#         raw_id = str(item.get('paper_id', ''))
#         clean_id = raw_id.replace('abs-', '').replace('arxiv:', '')
#         pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
        
#         results.append({
#             "title": item.get('title'),
#             "abstract": item.get('abstract'),
#             "authors": item.get('authors'),
#             "year": item.get('year'),
#             "pdf_url": pdf_url,
#             "score": float(distances[0][i])
#         })
#     return results

# @app.post("/agent_summarize")
# def agentic_summary(payload: SummarizeRequest):
#     """
#     The Core Agent:
#     1. Downloads PDF
#     2. Sends to Gemini
#     3. Returns Fun Summary
#     """
#     print(f"🤖 Agent is reading: {payload.title}...")
    
#     # 1. Extract Text
#     full_text = get_pdf_text(payload.pdf_url)
#     if not full_text:
#         raise HTTPException(status_code=400, detail="Could not download PDF. ArXiv might be blocking requests.")

#     # 2. The "Fun" Prompt
#     prompt = f"""
#     You are a super-smart but hilarious Science YouTuber (like Veritaseum or Kurzgesagt). 
#     I have a research paper titled "{payload.title}".
    
#     Here is the full text of the paper:
#     {full_text[:30000]} # Truncated for safety, though 1.5 Flash can handle much more
    
#     Your Goal: Explain this paper so it is FUN and easy to understand.
    
#     Format the output exactly like this HTML:
    
#     <h2>🚀 The Big Idea (TL;DR)</h2>
#     <p>[One simple paragraph explaining what problem they solved]</p>
    
#     <h2>🧠 How It Works (The "Magic")</h2>
#     <ul>
#         <li>[Bullet point 1]</li>
#         <li>[Bullet point 2]</li>
#     </ul>
    
#     <h2>🌍 Why Should You Care?</h2>
#     <p>[Real world analogy or application]</p>
    
#     <h2>🤓 The "Nerd" Details</h2>
#     <p>[A slightly more technical summary for when I want to sound smart]</p>
#     """

#     # 3. Generate
#     response = model.generate_content(prompt)
#     return {"html_content": response.text}

# # Serve the static frontend
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

# def get_pdf_text(url):
#     """
#     1. Checks if PDF exists locally in 'pdfs/' folder.
#     2. If yes -> Reads it.
#     3. If no -> Downloads it, SAVES it, then reads it.
#     """
#     try:
#         # 1. Generate a filename from the URL
#         # Example: https://arxiv.org/pdf/2107.10390.pdf -> 2107.10390.pdf
#         filename = url.split("/")[-1]
#         if not filename.endswith(".pdf"): 
#             filename += ".pdf"
            
#         local_path = os.path.join(PDF_CACHE_DIR, filename)

#         # 2. CHECK: Do we already have it?
#         if os.path.exists(local_path):
#             print(f"📂 Found local copy: {local_path}")
#             # Read from disk
#             reader = PdfReader(local_path)
#             text = ""
#             for page in reader.pages[:10]: # Read first 10 pages
#                 text += page.extract_text() + "\n"
#             return text

#         # 3. DOWNLOAD: If we don't have it, fetch it
#         print(f"📥 Downloading from ArXiv: {url}")
        
#         # Robust headers to look like a real browser
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#             'Referer': 'https://arxiv.org/'
#         }
        
#         response = requests.get(url, headers=headers, timeout=15)
        
#         if response.status_code == 200:
#             # SAVE to disk for next time
#             with open(local_path, "wb") as f:
#                 f.write(response.content)
#             print(f"✅ Saved to {local_path}")
            
#             # Read the newly saved file
#             reader = PdfReader(local_path)
#             text = ""
#             for page in reader.pages[:10]:
#                 text += page.extract_text() + "\n"
#             return text
#         else:
#             print(f"❌ Failed to download. Status: {response.status_code}")
#             return None

#     except Exception as e:
#         print(f"❌ Error processing PDF: {e}")
#         return None

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import json
import faiss
import numpy as np
import requests
import io
import re
import glob
import ssl
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import google.generativeai as genai

# --- 1. SSL FIX FOR MAC (Prevents download errors) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- 2. CONFIGURATION ---
# PASTE YOUR API KEY HERE
# 1. Force Python to find .env in the same folder as app.py
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("No GOOGLE_API_KEY found. Please check your .env file.")

PDF_CACHE_DIR = "pdfs"
INDEX_PATH = "index/papers.faiss"
METADATA_PATH = "index/metadata.json"

os.makedirs(PDF_CACHE_DIR, exist_ok=True)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. LOAD RESOURCES ---
search_index = None
search_model = None
paper_metadata = []

@app.on_event("startup")
def load_resources():
    global search_index, search_model, paper_metadata
    print("⏳ Loading AI Models...")
    search_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    search_index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        paper_metadata = json.load(f)
    print(f"System Ready! {len(paper_metadata)} papers loaded.")

# --- 4. HYBRID DOWNLOADER (Fixes 'File Not Found') ---
def get_pdf_text(url):
    try:
        clean_name = url.split("/")[-1]
        paper_id = clean_name.replace(".pdf", "")
        paper_id_core = paper_id.split('v')[0] # Remove version (v1)

        # A. Check Local
        search_pattern = os.path.join(PDF_CACHE_DIR, f"{paper_id_core}*.pdf")
        found_files = glob.glob(search_pattern)

        if found_files:
            print(f"📂 Found local: {found_files[0]}")
            reader = PdfReader(found_files[0])
            text = ""
            for i in range(min(len(reader.pages), 15)):
                text += reader.pages[i].extract_text() + "\n"
            return text

        # B. Download if missing
        print(f"Downloading: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Chrome/91.0)', 'Referer': 'https://arxiv.org/'}
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            save_path = os.path.join(PDF_CACHE_DIR, f"{paper_id}.pdf")
            with open(save_path, "wb") as f:
                f.write(response.content)
            
            reader = PdfReader(save_path)
            text = ""
            for i in range(min(len(reader.pages), 15)):
                text += reader.pages[i].extract_text() + "\n"
            return text
        return None

    except Exception as e:
        print(f" PDF Error: {e}")
        return None

# --- API ENDPOINTS ---

class SearchQuery(BaseModel):
    query: str

class SummarizeRequest(BaseModel):
    pdf_url: str
    title: str

@app.post("/search")
def search_papers(payload: SearchQuery):
    query_vector = search_model.encode([payload.query], normalize_embeddings=True).astype('float32')
    distances, indices = search_index.search(query_vector, k=10) 
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue
        item = paper_metadata[idx]
        raw_id = str(item.get('paper_id', ''))
        clean_id = raw_id.replace('abs-', '').replace('arxiv:', '')
        pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
        
        results.append({
            "title": item.get('title'),
            "abstract": item.get('abstract'),
            "authors": item.get('authors'),
            "year": item.get('year'),
            "pdf_url": pdf_url
        })
    return {"papers": results}

@app.post("/agent_summarize")
def agentic_summary(payload: SummarizeRequest):
    print(f"🤖 Reading: {payload.title}")
    
    full_text = get_pdf_text(payload.pdf_url)
    if not full_text:
        return {"html_content": "<h3>Error</h3><p>Could not download paper.</p>", "faq": [], "recommendations": []}

    # --- 5. ROBUST PROMPT WITH SEPARATOR ---
    prompt = f"""
    You are a quirky Science YouTuber. Explain this paper: "{payload.title}".
    
    Paper Text (Truncated):
    {full_text[:30000]}

    INSTRUCTIONS:
    1. Write a Fun Blog Post (Markdown). Use ## headings.
    2. Type strictly this separator: |||FAQ|||
    3. Write 3 pairs of Questions and Answers in this exact format:
       Q: [Question]
       A: [Answer]
    
    Example Output:
    # Title
    ## Big Idea
    (Content...)

    |||FAQ|||
    Q: Is this real-time?
    A: Yes, it runs at 30fps.
    Q: What data?
    A: Only ImageNet.
    """
    
    summary_markdown = ""
    faq_list = []
    
    try:
        response = model.generate_content(prompt)
        text_resp = response.text
        
        # --- 6. SPLIT BY SEPARATOR (Crash Proof) ---
        if "|||FAQ|||" in text_resp:
            parts = text_resp.split("|||FAQ|||")
            summary_markdown = parts[0].strip()
            
            # Parse Q&A
            raw_faq = parts[1].strip()
            # Regex to find Q: ... A: ... blocks
            matches = re.findall(r"Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)", raw_faq, re.DOTALL)
            for q, a in matches:
                faq_list.append({"question": q.strip(), "answer": a.strip()})
        else:
            summary_markdown = text_resp
            faq_list = [{"question": "AI generated no FAQ", "answer": "Reading the full text might help!"}]

    except Exception as e:
        summary_markdown = f"### Error\n{str(e)}"

    # --- 7. RECOMMENDATIONS ---
    recommendations = []
    try:
        query_vector = search_model.encode([payload.title], normalize_embeddings=True).astype('float32')
        distances, indices = search_index.search(query_vector, k=4)
        for idx in indices[0]:
            if idx == -1: continue
            item = paper_metadata[idx]
            if item.get('title') == payload.title: continue
            
            raw_id = str(item.get('paper_id', ''))
            clean_id = raw_id.replace('abs-', '').replace('arxiv:', '')
            recommendations.append({
                "title": item.get('title'),
                "year": item.get('year'),
                "pdf_url": f"https://arxiv.org/pdf/{clean_id}.pdf"
            })
    except:
        pass

    return {
        "html_content": summary_markdown, 
        "faq": faq_list,
        "recommendations": recommendations[:3]
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)