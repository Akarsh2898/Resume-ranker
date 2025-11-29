# AI-Powered Resume Ranker

Rank resumes for a job profile using NLP techniques (SpaCy + TF-IDF) and a simple Flask UI.

## Features
- Upload multiple PDF resumes and a job description
- Text extraction from PDFs
- SpaCy-based preprocessing (lemmatization, stopword removal)
- TF-IDF vectorization and cosine similarity
- Keyword-based scoring from job description
- Combined score and ranking with results table
- Downloadable HR report (CSV)

## Setup
1. Create/activate a Python environment (3.9+ recommended).
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. (Optional but recommended) Install SpaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
   The app will fallback to a basic tokenizer if the model is missing.

## Run
```bash
python app.py
```
Open http://127.0.0.1:5000/ and upload PDF resumes along with a job description.

## Scoring Details
- Base score: cosine similarity between resume and job description TF-IDF vectors
- Keyword score: proportion of top job-description keywords appearing in the resume
- Final score: `0.7 * base_score + 0.3 * keyword_score`

## Deliverables
- Flask app (`app.py`)
- Scoring algorithm (`resume_ranker/processor.py`)
- UI templates (`templates/`), styling (`static/`)
- Sample output placeholder: generate via the UI and click “Download HR Report”

## Notes
- PDF text extraction uses `PyPDF2`. Complex or scanned PDFs may need OCR.
- For production, consider model caching, file size limits, and robust error handling.