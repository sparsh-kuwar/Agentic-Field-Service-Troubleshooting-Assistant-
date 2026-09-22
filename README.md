# Agentic Field-Service Troubleshooting Assistant

An AI-powered field-service troubleshooting assistant that uses **Retrieval-Augmented Generation (RAG)**, **FAISS semantic search**, **LLMs**, and a lightweight **multi-agent workflow** to help technicians diagnose equipment problems from service manuals.

The system is designed to reduce unsupported AI answers by grounding recommendations in retrieved manual content and automatically escalating problems when the retrieved evidence is not sufficiently relevant.

---

## 🚀 Project Overview

Field technicians often need to troubleshoot equipment using large technical manuals. Searching these manuals manually can be slow, especially when the technician only has a short description of the fault.

This project builds an AI assistant that takes a technician's problem description and:

1. Searches the equipment manual semantically.
2. Retrieves the most relevant sections.
3. Checks whether the retrieved evidence is sufficiently relevant.
4. Escalates to a human service engineer when evidence is weak.
5. If the evidence is strong enough, runs three specialized AI agents:
   - Diagnosis Agent
   - Parts Recommendation Agent
   - Next-Best-Action Agent
6. Produces a grounded troubleshooting recommendation.

---

## 🏗️ Architecture

```text
                         Equipment Manual PDF
                                  |
                                  v
                         PDF Document Loader
                                  |
                                  v
                         Recursive Chunking
                                  |
                                  v
                            Embeddings
                                  |
                                  v
                         FAISS Vector Store
                                  |
                         Persisted Locally
                                  |
                                  v
                    +-------------------------+
                    |   Technician Problem    |
                    +-------------------------+
                                  |
                                  v
                       Semantic Retrieval
                                  |
                                  v
                       Relevance Score
                                  |
                    +-------------+-------------+
                    |                           |
              Score < 0.25                Score >= 0.25
                    |                           |
                    v                           v
             Human Escalation            Agent Workflow
                                                |
                           +--------------------+--------------------+
                           |                    |                    |
                           v                    v                    v
                    Diagnosis Agent       Parts Agent       Next-Action Agent
                           |                    |                    |
                           +--------------------+--------------------+
                                                |
                                                v
                                  Troubleshooting Recommendation
```

---

# 🧠 Core Technologies

| Technology | Purpose |
|---|---|
| Python | Application development |
| LangChain | LLM and workflow orchestration |
| LangChain OpenAI | OpenAI LLM integration |
| PyPDF | PDF document loading |
| Sentence Transformers | Text embeddings |
| HuggingFace Embeddings | Embedding integration with LangChain |
| FAISS | Vector similarity search |
| OpenAI LLM | Reasoning and response generation |
| python-dotenv | Secure environment variable loading |

---

# 🔍 How the System Works

## 1. PDF Document Loading

The equipment manual is stored under:

```text
data/manuals/
```

The application uses `PyPDFLoader` to extract text from the PDF.

Example:

```text
X200_Industrial_Water_Pump_Manual.pdf
```

The manual contains equipment specifications, fault codes, troubleshooting procedures, service parts, and safety instructions.

---

## 2. Document Chunking

Large documents are not sent directly to the LLM.

Instead, the manual is divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
```

### Why chunking?

A complete manual may contain hundreds of pages. Sending the entire manual to an LLM for every question would be inefficient and expensive.

Chunking allows the system to search smaller sections of the manual.

### Chunk overlap

The 150-character overlap helps preserve context between neighboring chunks.

For example:

```text
Chunk 1:
... troubleshooting steps for low pressure ...

Chunk 2:
... final troubleshooting step + relevant parts ...
```

Without overlap, important information near a chunk boundary could be lost.

---

# 🧮 3. Text Embeddings

Each document chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model produces a **384-dimensional vector** representation.

Conceptually:

```text
"pump pressure is low"
            ↓
[0.12, -0.04, 0.73, ... 384 values]
```

The vector represents the semantic meaning of the text.

This allows the system to compare meanings rather than relying only on exact keyword matches.

---

# 🗄️ 4. FAISS Vector Store

The embeddings are stored in a FAISS vector index.

FAISS allows the application to perform fast similarity searches.

Instead of searching for exact words:

```text
"low pressure"
```

the system can retrieve information related to:

```text
"reduced discharge pressure and decreased water flow"
```

even if the manual uses different wording.

---

# 💾 5. Persistent Vector Store

The FAISS index is saved locally:

```text
vectorstore/
├── index.faiss
└── index.pkl
```

This prevents the application from rebuilding the embeddings every time the assistant starts.

### Initial setup

```text
PDF
 ↓
Chunks
 ↓
Embeddings
 ↓
FAISS
 ↓
Save index
```

### Every subsequent query

```text
Question
 ↓
Load existing FAISS
 ↓
Semantic search
 ↓
Retrieve relevant chunks
```

The `vectorstore/` directory is excluded from Git because it is a generated artifact and can be recreated locally.

---

# 🔎 6. Semantic Retrieval

When a technician enters a problem, FAISS retrieves the top three relevant chunks.

Example input:

```text
The pump pressure is very low and the water flow has decreased.
```

The system retrieves manual sections related to:

- low discharge pressure
- inlet water supply
- inlet filter
- discharge valve
- impeller condition

The application uses:

```python
similarity_search_with_relevance_scores()
```

to retrieve both documents and relevance scores.

---

# 🛡️ 7. Confidence Gate and Human Escalation

One of the important safety mechanisms is the retrieval-confidence gate.

The system checks the strongest relevance score before sending the retrieved information to the agent workflow.

Current project threshold:

```python
CONFIDENCE_THRESHOLD = 0.25
```

The logic is:

```text
Retrieved evidence
       |
       v
Best relevance score
       |
       +---- score < 0.25 ----> Human escalation
       |
       +---- score >= 0.25 ---> Agent workflow
```

### Why?

An LLM can generate a plausible answer even when the knowledge base does not contain useful information.

The confidence gate reduces this risk by preventing generation when retrieval evidence is too weak.

Example:

```text
Question:
"What is the recommended tire pressure for a car?"

Manual:
X200 industrial water pump manual

Result:
Insufficient relevant evidence

Action:
Escalate to human service engineer
```

The threshold is a configurable project-level value and should be calibrated against a larger labeled evaluation set before production deployment.

---

# 🤖 8. Agentic Workflow

Once the retrieval evidence passes the confidence gate, the application uses three specialized agents.

## Diagnosis Agent

The Diagnosis Agent identifies likely causes using only the retrieved manual context.

Input:

```text
Technician problem
+
Retrieved manual context
```

Output:

```text
Likely causes
+
Supporting evidence
```

---

## Parts Recommendation Agent

The Parts Agent receives:

```text
Manual context
+
Diagnosis
```

It identifies relevant service parts mentioned in the manual.

Example:

```text
Inlet filter
Impeller
Discharge valve
```

The agent is explicitly instructed not to invent parts that are not present in the manual.

---

## Next-Best-Action Agent

The final agent receives:

```text
Manual context
+
Diagnosis
+
Relevant parts
```

It generates an ordered troubleshooting sequence.

Example:

```text
1. Verify inlet water supply is at least 40 L/min.
2. Inspect the inlet filter.
3. Confirm the discharge valve is fully open.
4. Recheck discharge pressure.
5. Inspect the impeller if pressure remains below 3.0 bar.
```

Safety instructions from the manual are also included when relevant.

---

# 🔗 Complete RAG + Agent Pipeline

The complete flow is:

```text
Technician Problem
       |
       v
Semantic Search
       |
       v
Top 3 Manual Chunks
       |
       v
Relevance Score
       |
       +--------------------+
       |                    |
     Weak                 Strong
       |                    |
       v                    v
Human Escalation       Diagnosis Agent
                            |
                            v
                       Parts Agent
                            |
                            v
                    Next-Action Agent
                            |
                            v
                  Final Recommendation
```

---

# 📁 Project Structure

```text
field-service-troubleshooting-agent/
│
├── data/
│   └── manuals/
│       └── X200_Industrial_Water_Pump_Manual.pdf
│
├── src/
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── agents.py
│   └── rag.py
│
├── vectorstore/
│   └── generated FAISS index
│
├── tests/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

### File responsibilities

| File | Responsibility |
|---|---|
| `ingestion.py` | Loads and chunks equipment manuals |
| `embeddings.py` | Generates text embeddings |
| `vector_store.py` | Creates, saves, and loads FAISS |
| `agents.py` | Contains Diagnosis, Parts, and Next-Action agents |
| `rag.py` | Main application and orchestration |
| `data/manuals/` | Equipment manuals |
| `vectorstore/` | Generated FAISS index |

---

# ⚙️ Setup

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd field-service-troubleshooting-agent
```

## 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been generated yet, install:

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters
pip install langchain-huggingface sentence-transformers faiss-cpu pypdf python-dotenv
```

---

# 🔐 API Key Configuration

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

Never commit `.env` to GitHub.

The repository `.gitignore` contains:

```text
venv/
.env
__pycache__/
vectorstore/
```

---

# ▶️ Running the Project

## Step 1 — Build the FAISS index

Run once after adding or changing manuals:

```bash
python src/vector_store.py
```

This creates:

```text
vectorstore/
├── index.faiss
└── index.pkl
```

---

## Step 2 — Start the assistant

```bash
python src/rag.py
```

Example:

```text
================================
FIELD SERVICE TROUBLESHOOTING
AI ASSISTANT
================================

Describe the equipment problem:
> The pump pressure is very low and the water flow has decreased.
```

The system then performs retrieval, confidence checking, and agent execution.

---

# 🧪 Example Output

For a low-pressure pump problem, the system can identify:

### Diagnosis

```text
- Insufficient inlet water supply
- Blocked inlet filter
- Discharge valve not fully open
- Worn or damaged impeller
```

### Relevant Parts

```text
- Inlet filter
- Impeller
- Discharge valve
```

### Next-Best Actions

```text
1. Verify inlet water supply.
2. Inspect inlet filter.
3. Confirm discharge valve position.
4. Recheck discharge pressure.
5. Inspect the impeller if required.
```

### Safety

```text
Disconnect electrical power before opening
or inspecting rotating components.
```

---

# 🧪 Testing Strategy

The project should be tested with three categories of queries.

## 1. In-scope queries

Questions that clearly relate to the equipment manual.

Example:

```text
The pump pressure is very low.
```

Expected:

```text
Retrieve evidence
→ Pass confidence gate
→ Run agents
→ Generate troubleshooting steps
```

## 2. Out-of-scope queries

Questions unrelated to the equipment.

Example:

```text
What is the recommended tire pressure for a car?
```

Expected:

```text
Low retrieval relevance
→ Human escalation
```

## 3. Ambiguous queries

Example:

```text
The machine is behaving strangely.
```

Expected behavior depends on retrieved evidence. If the manual does not provide sufficient support, the system should escalate rather than invent a diagnosis.

---

# 🔒 Safety and Grounding

The project is designed around several safeguards:

### Manual-grounded generation

Agents are instructed to use only retrieved manual information.

### No unsupported parts

The Parts Agent should only recommend parts found in the manual.

### Confidence gate

Weak retrieval results trigger escalation.

### Human-in-the-loop

The system does not attempt to solve unsupported problems autonomously.

### Safety instructions

Relevant safety instructions from the manual are preserved in the recommended actions.

---

# 🚧 Current Limitations

This is a portfolio/research implementation rather than a production field-service platform.

Current limitations include:

- The confidence threshold is manually configured.
- Evaluation data is limited.
- Only a sample equipment manual is currently included.
- The system uses sequential LLM calls, which can increase latency and API cost.
- No authentication or user management is implemented.
- No production database is included.
- No frontend is currently included.
- The REST API layer has not yet been added.
- Production deployment and observability are not included.

---

# 🔮 Future Improvements

Potential production improvements:

1. Add FastAPI REST endpoints.
2. Add a technician web/mobile interface.
3. Store troubleshooting history.
4. Add multiple equipment manuals.
5. Add metadata filtering by equipment model.
6. Calibrate the confidence threshold using labeled test data.
7. Add structured JSON/Pydantic outputs.
8. Add evaluation metrics such as retrieval accuracy and grounded-answer rate.
9. Add LangSmith or another observability layer.
10. Add authentication and role-based access.
11. Add monitoring and error tracking.
12. Add automated tests and CI/CD.
13. Support incremental document indexing when manuals change.

---

# 💡 Key Concepts Demonstrated

This project demonstrates practical understanding of:

- Retrieval-Augmented Generation (RAG)
- Semantic search
- Text embeddings
- Vector databases
- FAISS
- Document chunking
- LangChain
- LLM orchestration
- Prompt engineering
- Agentic workflows
- Human-in-the-loop systems
- Confidence-based escalation
- Grounded generation
- Python application architecture

---

# 🎯 Interview Explanation

A concise way to explain the project:

> "I built an agentic field-service troubleshooting assistant that uses RAG to ground an LLM in equipment service manuals. I first load and chunk the manuals, generate embeddings using Sentence Transformers, and store them in FAISS for semantic retrieval. When a technician describes a problem, the system retrieves the most relevant manual sections and checks their relevance score. If the evidence is insufficient, it escalates to a human service engineer instead of generating an unsupported answer. For sufficiently relevant queries, I use three specialized agents for diagnosis, parts recommendation, and next-best-action planning. This creates a grounded troubleshooting workflow rather than relying on the LLM's pretrained knowledge alone."

---

# 👨‍💻 Author

**Sparsh Kuwar**

Computer Science Engineering  
JSPM Rajarshi Shahu College of Engineering

