# DeepRead AI: Generative Research Assistant

**DeepRead AI** is a hybrid recommender system and agentic research assistant designed to make academic literature accessible, engaging, and discoverable.

It combines **Dense Vector Retrieval (FAISS)** for semantic search with **KNN** for discovery, effectively solving the "Cold Start" problem in research while providing an **Agentic RAG** interface to summarize and explain complex PDFs.

---

##  Key Features

###  1. Hybrid Search Engine
* **Semantic Search:** Uses **Sentence-Transformers** (BERT-based) to understand the *meaning* of a query, not just keyword matching.
* **Discovery Engine:** Implements the **KNN Algorithm** to analyze the retrieved papers and surface hidden topics (e.g., *"People reading about Neural Networks also explore Optimization"*).

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
| **Discovery Layer** | `mlxtend` (KNN) | **KNN:** Mining frequent itemsets to calculate Association Rules . |
| **Generation Layer** | `Google Gemini API` | **GenRec (Generative Recommendation):** Using LLMs to explain *why* an item is relevant and summarize it. |

---

## Installation & Setup

### Prerequisites
* Python 3.9+
* A Google Cloud API Key (for Gemini)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yashwanth008/CMPE-256_Final_project.git
cd CMPE-256_Final_project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages include:
- `fastapi` - Web framework for building APIs
- `uvicorn` - ASGI server
- `sentence-transformers` - For semantic embeddings
- `faiss-cpu` - Vector similarity search
- `google-generativeai` - Google Gemini API
- `pypdf` - PDF text extraction
- `python-dotenv` - Environment variable management
- `spacy` - Named Entity Recognition (NER)
- `mlxtend` - KNN algorithm for association rules

### Step 3: Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### Step 4: Configure Google API Key

**IMPORTANT**: You need a Google Gemini API key to use the AI summarization features.

#### 4.1 Get Your API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

#### 4.2 Set Up Environment Variables
Create a `.env` file in the project root directory:

```bash
# On Windows (Command Prompt)
echo GOOGLE_API_KEY=your_api_key_here > .env

# On macOS/Linux
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

Or manually create a file named `.env` with the following content:
```plaintext
GOOGLE_API_KEY=your_actual_api_key_here
```

**Replace `your_actual_api_key_here` with your actual Google API key.**

**Security Note**:
- Never commit your `.env` file to Git (it's already in `.gitignore`)
- Keep your API key private and do not share it publicly

### Step 5: Prepare the Data
Ensure you have the FAISS index and metadata files in the `index/` folder:
- `index/papers.faiss` - Vector embeddings index
- `index/metadata.json` - Paper metadata (titles, abstracts, authors, etc.)

### Step 6: Run the Application
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

The application will start on `http://127.0.0.1:8000`

### Step 7: Access the Web Interface
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## Usage Guide

### Search for Papers
1. Enter a search query in the search box (e.g., "neural networks", "transformer models")
2. The system supports advanced queries:
   - **By topic**: "attention mechanisms in NLP"
   - **By author**: "papers by Yoshua Bengio"
   - **By year**: "deep learning 2017"
   - **Combined**: "transformers by Vaswani 2017"
3. Results are ranked using semantic similarity (not just keyword matching)

### Read AI-Generated Summaries
1. Click on any paper from the search results
2. The system will:
   - Download the PDF (if not cached locally)
   - Extract text from the first 15 pages
   - Generate an engaging summary using Google Gemini
   - Provide a FAQ section with beginner and advanced questions
   - Recommend 3 related papers to explore next

### Discovery with Association Rules
The system analyzes reading patterns to discover hidden topics:
- Papers frequently read together are grouped
- Association rules reveal topic relationships
- Example: "Readers of Attention papers also explore Optimization techniques"

---

## Project Structure

```
CMPE-256_Final_project/
├── app.py                  # Main FastAPI application
├── static/                 # Frontend HTML/CSS/JS files
├── index/
│   ├── papers.faiss       # FAISS vector index
│   └── metadata.json      # Paper metadata
├── pdfs/                   # Downloaded PDF cache
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## API Endpoints

### POST `/search`
Search for papers using semantic similarity.

**Request Body:**
```json
{
  "query": "transformer neural networks"
}
```

**Response:**
```json
{
  "papers": [
    {
      "title": "Attention Is All You Need",
      "abstract": "...",
      "authors": "Vaswani et al.",
      "year": "2017",
      "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"
    }
  ],
  "extracted": {
    "topic": "transformer neural networks",
    "author": null,
    "year": null
  }
}
```

### POST `/agent_summarize`
Generate an AI summary of a research paper.

**Request Body:**
```json
{
  "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
  "title": "Attention Is All You Need"
}
```

**Response:**
```json
{
  "html_content": "## The Problem\n...",
  "faq": [
    {
      "question": "What is a Transformer?",
      "answer": "..."
    }
  ],
  "recommendations": [
    {
      "title": "BERT: Pre-training of Deep Bidirectional Transformers",
      "year": "2018",
      "pdf_url": "..."
    }
  ]
}
```

---

## Troubleshooting

### Error: "No GOOGLE_API_KEY found"
**Solution**: Make sure you created a `.env` file in the project root with your API key:
```plaintext
GOOGLE_API_KEY=your_actual_api_key_here
```

### Error: "Could not download PDF"
**Possible causes**:
1. ArXiv is blocking requests (rate limiting)
2. The PDF doesn't exist at that URL
3. Network connectivity issues

**Solution**:
- Manually download the PDF from ArXiv
- Save it to the `pdfs/` folder with the paper ID as filename (e.g., `1706.03762.pdf`)
- The system will automatically use the local copy

### Error: "OSError: [WinError 10054] connection was forcibly closed"
**Solution**: This is a Windows-specific SSL issue. The code already includes a fix at lines 279-284 in [app.py](app.py#L279-L284).

### spaCy model not found
**Solution**: Download the language model:
```bash
python -m spacy download en_core_web_sm
```

### FAISS index or metadata missing
**Solution**: Ensure the following files exist:
- `index/papers.faiss`
- `index/metadata.json`

These files should be provided with the repository or generated during the indexing phase.

---

## Technical Details

### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Purpose**: Convert text to dense vectors for semantic search

### Vector Search
- **Engine**: FAISS (Facebook AI Similarity Search)
- **Index Type**: Flat L2 (exact nearest neighbor)
- **Top-K**: Returns top 50 candidates, filters to top 10

### LLM for Summarization
- **Model**: Google Gemini 2.5 Flash Lite
- **Context Window**: 1M tokens (can read entire papers)
- **Features**: Generates summaries, FAQs, and explanations

### Named Entity Recognition
- **Model**: spaCy `en_core_web_sm`
- **Purpose**: Extract author names and years from search queries

---

## Features in Detail

### Cold Start Solution
Traditional recommender systems fail when there's no user history. DeepRead AI solves this by:
1. **Semantic Search**: Understands query intent, not just keywords
2. **Association Rules**: Analyzes global reading patterns to suggest topics
3. **Content-Based Recommendations**: Uses paper similarity for suggestions

### Smart PDF Caching
- Checks local `pdfs/` folder before downloading
- Prevents redundant network calls
- Respects ArXiv's rate limits
- Privacy-focused (local processing)

### Query Understanding
The system intelligently parses queries:
- **"papers by Bengio"** → Filters by author
- **"deep learning 2015"** → Filters by topic + year
- **"attention mechanisms"** → Pure semantic search

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **ArXiv**: For providing open access to research papers
- **Google Gemini**: For powerful LLM capabilities
- **FAISS**: For efficient similarity search
- **Sentence Transformers**: For semantic embeddings
- **spaCy**: For NER capabilities

---

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/yashwanth008/CMPE-256_Final_project/issues).

---

**Built with by the Venkata Yashwanth Paladugu, Prachi Gupta and Aniket Anil Naik**
