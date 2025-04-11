# Accenture-Hackathon
# HireSense

HireSense is an AI-powered, agent-based recruitment platform that automates the end-to-end hiring workflow—from optimizing job descriptions to processing candidate CVs, ensuring fairness, explainability, and efficiency throughout the recruitment process.

## Features

- **JD Optimizer:**  
  Extracts and cleans raw job descriptions with NLP, optimizing for clarity and inclusivity.

- **CV Parser & Grader:**  
  Processes candidate CVs (PDF/TXT) using semantic embeddings and keyword matching to score relevance against the optimized JD.

- **Bias & Fairness:**  
  Detects biased language and anonymizes texts to ensure fairness across job descriptions and CVs.

- **Persona Fit Analysis:**  
  Evaluates soft skills and cultural compatibility via sentiment analysis and soft-skills keyword counts.

- **Explainability:**  
  Generates human-readable explanations of candidate scores using SHAP-based insights.

- **Recruiter Feedback:**  
  Incorporates recruiter feedback to adjust candidate scores in real time.

- **SQLite Memory:**  
  Provides a central, persistent database to store and query recruitment data.

- **Coordinator (Supervisor):**  
  Orchestrates all agents in sequence, ensuring that each module’s output becomes the input of the next and ultimately presents the final selected candidates.

## Tech Stack

- **Language:** Python 3.x  
- **NLP & Text Processing:** spaCy (NER, tokenization, lemmatization), PyPDF2 (PDF extraction)  
- **Semantic Embeddings:** Sentence Transformers (all‑MiniLM‑L6‑v2) using PyTorch (set `TRANSFORMERS_NO_TF=1`)  
- **Explainability:** SHAP for interpretable model explanations  
- **Data Persistence:** SQLite (via Python's `sqlite3`)  
- **Dashboard (Optional):** Streamlit for real-time interactive UI  
- **Orchestration:** CLI tools and a supervisor script to run the full pipeline

## Directory Structure

