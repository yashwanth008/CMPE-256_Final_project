# DeepRead AI: Generative Research Assistant

**DeepRead AI** is a hybrid recommender system and agentic research assistant designed to make academic literature accessible, engaging, and discoverable.

It combines **Dense Vector Retrieval (FAISS)** for semantic search with **Association Rule Mining (Apriori)** for discovery, effectively solving the "Cold Start" problem in research while providing an **Agentic RAG** interface to summarize and explain complex PDFs.

---

##  Key Features

###  1. Hybrid Search Engine
* **Semantic Search:** Uses **Sentence-Transformers** (BERT-based) to understand the *meaning* of a query, not just keyword matching.
* **Discovery Engine:** Implements the **Apriori Algorithm** to analyze the retrieved papers and surface hidden topics (e.g., *"People reading about Neural Networks also explore Optimization"*).

### 2. Agentic RAG Summarizer
* **"Fun Mode" Summaries:** Transforms dry academic text into engaging, blog-style content using **Google Gemini 2.5 Flash-Lite**.
* **Robust Parsing:** Uses a custom separator strategy to ensure crash-proof JSON/Markdown extraction.
* **Interactive Chat:** Allows users to ask specific questions about the paper ("What was the learning rate?") with instant, grounded answers.

### 3. Smart Local Caching
* **Hybrid Downloader:** Automatically checks a local `pdfs/` folder first. If the file is missing, it auto-downloads it from ArXiv to prevent redundant network calls and IP blocks.
* **Privacy First:** Papers are processed locally after download.

### 4. Item-to-Item Recommendations
* **Contextual Suggestions:** At the end of every summary, the system uses vector similarity to recommend the next 3 most relevant papers to read, creating an infinite learning loop.

---

## Technical Architecture (RecSys Stack)

This project aligns closely with modern Recommender Systems architectures:

| Component | Technology | RecSys Concept |
| :--- | :--- | :--- |
| **Embedding Layer** | `all-MiniLM-L6-v2` | **Content-Based Filtering:** Converting unstructured text into dense vector space. |
| **Retrieval Layer** | `FAISS` (Facebook AI Similarity Search) | **Candidate Generation:** Approximate Nearest Neighbor (ANN) search for low-latency retrieval. |
| **Discovery Layer** | `mlxtend` (Apriori) | **Market Basket Analysis:** Mining frequent itemsets to calculate Association Rules (Support/Confidence). |
| **Generation Layer** | `Google Gemini API` | **GenRec (Generative Recommendation):** Using LLMs to explain *why* an item is relevant and summarize it. |

---

## Installation & Setup

### Prerequisites
* Python 3.9+
* A Google Cloud API Key (for Gemini)

### 1. Clone the Repository
```bash
git clone https://github.com/yashwanth008/CMPE-256_Final_project.git
cd DeepRead-ai
