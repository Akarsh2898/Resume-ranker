import io
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
try:
    from sentence_transformers import SentenceTransformer
    SBERT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception:
    SBERT_MODEL = None
from PyPDF2 import PdfReader


def _load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        # Fallback to a blank English model (tokenizer only)
        return spacy.blank("en")


NLP = _load_nlp()


def extract_text_from_pdf(file_storage) -> str:
    reader = PdfReader(file_storage.stream)
    texts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        texts.append(text)
    return "\n".join(texts)

def extract_text_from_pdf_path(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        texts.append(text)
    return "\n".join(texts)


HARD_SKILL_KEYWORDS = {
    "python","java","c++","sql","tableau","power bi","excel","machine learning","deep learning","nlp","flask","django","pandas","numpy","scikit-learn","tensorflow","pytorch","keras","spark","aws","azure","gcp","docker","kubernetes","git","linux","bash","rest","api","html","css","javascript","react","node","data analysis","data engineering","feature engineering","xgboost","lightgbm","random forest","logistic regression","classification","regression","clustering","time series"
}
SOFT_SKILL_KEYWORDS = {
    "communication","teamwork","leadership","problem solving","critical thinking","collaboration","stakeholder management","presentation","mentoring","ownership","adaptability","creativity","decision making","project management","time management"
}


def extract_skills(text: str) -> Dict[str, List[str]]:
    t = text.lower()
    hard = sorted({k for k in HARD_SKILL_KEYWORDS if k in t})
    soft = sorted({k for k in SOFT_SKILL_KEYWORDS if k in t})
    return {"hard_skills": hard, "soft_skills": soft}


def extract_experience_education(text: str) -> Tuple[float, str, str]:
    import re
    t = text
    yrs = 0.0
    for m in re.findall(r"(\d+\.?\d*)\s+years?", t, flags=re.I):
        try:
            yrs = max(yrs, float(m))
        except Exception:
            pass
    title = ""
    title_candidates = re.findall(r"\b([A-Z][A-Za-z/&\- ]{2,}\b(?:Engineer|Developer|Scientist|Manager|Analyst|Lead|Consultant))", t)
    if title_candidates:
        title = title_candidates[0]
    edu = ""
    if re.search(r"Bachelor|B\.?Tech|BSc|BE", t, flags=re.I):
        edu = "Bachelors"
    if re.search(r"Master|M\.?Tech|MSc|MS", t, flags=re.I):
        edu = "Masters"
    if re.search(r"PhD|Doctorate", t, flags=re.I):
        edu = "PhD"
    return yrs, title, edu


def preprocess_text(text: str) -> List[str]:
    doc = NLP(text)
    tokens = []
    for token in doc:
        if not token.is_alpha:
            continue
        lemma = token.lemma_.lower().strip() if token.lemma_ else token.text.lower().strip()
        if not lemma or lemma in STOP_WORDS or len(lemma) < 2:
            continue
        tokens.append(lemma)
    return tokens


def _extract_job_keywords(job_tokens: List[str], top_k: int = 25) -> List[str]:
    counts = Counter(job_tokens)
    # Filter to moderately informative tokens
    filtered = [(t, c) for t, c in counts.items() if len(t) > 2]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in filtered[:top_k]]


def _vectorize_corpus(docs: List[List[str]]) -> Tuple[np.ndarray, TfidfVectorizer]:
    # TfidfVectorizer expects raw strings, so join tokens
    raw_docs = [" ".join(tokens) for tokens in docs]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(raw_docs)
    return tfidf, vectorizer


def _sbert_similarity(job_text: str, resume_texts: List[str]) -> List[float]:
    if SBERT_MODEL is None:
        return []
    texts = [job_text] + resume_texts
    emb = SBERT_MODEL.encode(texts, normalize_embeddings=True)
    job_vec = emb[0]
    cand = emb[1:]
    sims = (cand @ job_vec).tolist()
    return sims


def _keyword_match_score(candidate_tokens: List[str], job_keywords: List[str]) -> Tuple[float, List[str]]:
    cand_counts = Counter(candidate_tokens)
    matched = [kw for kw in job_keywords if cand_counts.get(kw, 0) > 0]
    # Simple ratio of matched keywords
    score = len(matched) / max(len(job_keywords), 1)
    return float(score), matched


def _generate_improvements(candidate_tokens: List[str], job_keywords: List[str], base_score: float, kw_score: float, matched: List[str]) -> List[str]:
    suggestions: List[str] = []
    cand_set = set(candidate_tokens)
    missing = [kw for kw in job_keywords if kw not in cand_set]

    if missing:
        top_missing = ", ".join(missing[:8])
        suggestions.append(f"Consider including or highlighting: {top_missing}")

    if base_score < 0.4:
        suggestions.append("Tailor your summary to the job description; emphasize relevant responsibilities.")

    if kw_score < 0.3:
        suggestions.append("Add specific skills/tools from the JD and show hands-on usage.")

    if len(matched) < max(3, int(0.3 * len(job_keywords))):
        suggestions.append("Quantify achievements with metrics (impact, % improvements, time/cost savings).")

    if not suggestions:
        suggestions.append("Resume aligns well; consider minor refinements and clearer outcomes.")

    return suggestions


def _component_scores(resume_text: str, resume_tokens: List[str], job_text: str) -> Tuple[float, float, float]:
    sk = extract_skills(resume_text)
    jt = job_text.lower()
    matched = [s for s in sk["hard_skills"] if s in jt]
    skill_score = len(matched) / max(len(sk["hard_skills"]) or 1, 1)
    yrs, _, edu = extract_experience_education(resume_text)
    exp_score = min(yrs / 10.0, 1.0)
    edu_score = 0.0
    if edu == "Bachelors":
        edu_score = 0.5
    elif edu == "Masters":
        edu_score = 0.8
    elif edu == "PhD":
        edu_score = 1.0
    return skill_score, exp_score, edu_score


def rank_resumes(resume_files, job_desc_text: str, weights: Dict[str, float] | None = None, engine: str | None = None) -> Tuple[List[Dict], Dict]:
    # Extract and preprocess
    resume_texts = []
    resume_tokens_list = []
    filenames = []

    for f in resume_files:
        filenames.append(f.filename)
        text = extract_text_from_pdf(f)
        resume_texts.append(text)
        resume_tokens_list.append(preprocess_text(text))
        yrs, title, edu = extract_experience_education(text)
        sk = extract_skills(text)
        resume_texts[-1] = text

    job_tokens = preprocess_text(job_desc_text)
    job_keywords = _extract_job_keywords(job_tokens)

    # Vectorize job + resumes together
    use_sbert = (engine or "auto") != "tfidf"
    sbert_sims = _sbert_similarity(job_desc_text, resume_texts) if use_sbert else []
    if sbert_sims:
        cos_scores = np.array(sbert_sims)
    else:
        corpus_tokens = [job_tokens] + resume_tokens_list
        tfidf, _ = _vectorize_corpus(corpus_tokens)
        job_vec = tfidf[0]
        resume_vecs = tfidf[1:]
        cos_scores = cosine_similarity(resume_vecs, job_vec).reshape(-1)

    # Keyword match scores
    results = []
    for i, tokens in enumerate(resume_tokens_list):
        kw_score, matched = _keyword_match_score(tokens, job_keywords)
        base_score = float(cos_scores[i])
        if weights:
            ss, es, ds = _component_scores(resume_texts[i], tokens, job_desc_text)
            comp = weights.get("skills", 0.0) * ss + weights.get("experience", 0.0) * es + weights.get("education", 0.0) * ds
            final_score = 0.4 * base_score + 0.6 * comp
        else:
            ss = es = ds = 0.0
            final_score = 0.7 * base_score + 0.3 * kw_score
        improvements = _generate_improvements(tokens, job_keywords, base_score, kw_score, matched)
        yrs, title, edu = extract_experience_education(resume_texts[i])
        sk = extract_skills(resume_texts[i])
        cand_missing = [kw for kw in job_keywords if kw not in set(tokens)]
        results.append({
            "filename": filenames[i],
            "base_score": round(base_score, 4),
            "keyword_score": round(kw_score, 4),
            "final_score": round(final_score, 4),
            "matched_keywords": matched,
            "missing_keywords": cand_missing,
            "improvements": improvements,
            "years_experience": yrs,
            "last_job_title": title,
            "education_level": edu,
            "hard_skills": sk["hard_skills"],
            "soft_skills": sk["soft_skills"],
            "skill_score": round(ss, 4),
            "experience_score": round(es, 4),
            "education_score": round(ds, 4),
        })

    # Sort by final score descending
    results.sort(key=lambda r: r["final_score"], reverse=True)

    skill_counter = Counter()
    for r in results:
        for s in r["hard_skills"]:
            skill_counter[s] += 1
    keyword_summary = {
        "job_keywords": job_keywords,
        "total_resumes": len(results),
        "top_skills": [k for k, _ in skill_counter.most_common(10)],
    }

    return results, keyword_summary


def generate_csv_report(results: List[Dict], job_title: str) -> Tuple[bytes, str]:
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Job Title", job_title])
    writer.writerow([])
    writer.writerow(["Candidate", "CosineSimilarity", "KeywordMatchScore", "FinalScore", "MatchedKeywords", "TopImprovementAspects"])
    for r in results:
        writer.writerow([
            r["filename"],
            r["base_score"],
            r["keyword_score"],
            r["final_score"],
            "; ".join(r.get("matched_keywords", [])),
            "; ".join(r.get("improvements", [])[:3]),
        ])
    csv_bytes = output.getvalue().encode("utf-8")
    filename = f"Resume_Rankings_{job_title.replace(' ', '_')}.csv"
    return csv_bytes, filename


def generate_combined_csv_report(multi_payload: Dict[str, Dict]) -> Tuple[bytes, str]:
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Role", "Candidate", "CosineSimilarity", "KeywordMatchScore", "FinalScore", "YearsExperience", "Education", "MatchedKeywords", "TopImprovementAspects"])
    for role_title, payload in multi_payload.items():
        results = payload.get("results", [])
        for r in results:
            writer.writerow([
                role_title,
                r.get("filename"),
                r.get("base_score"),
                r.get("keyword_score"),
                r.get("final_score"),
                r.get("years_experience"),
                r.get("education_level"),
                "; ".join(r.get("matched_keywords", [])),
                "; ".join(r.get("improvements", [])[:3]),
            ])
    csv_bytes = output.getvalue().encode("utf-8")
    filename = "Resume_Rankings_All_Roles.csv"
    return csv_bytes, filename


def rank_texts(resume_text_items: List[Tuple[str, str]], job_desc_text: str, weights: Dict[str, float] | None = None, engine: str | None = None) -> Tuple[List[Dict], Dict]:
    """
    Rank resumes provided as raw text strings instead of PDF files.
    Each item is (filename, text).
    """
    filenames = [fn for fn, _ in resume_text_items]
    resume_texts = [txt for _, txt in resume_text_items]
    resume_tokens_list = [preprocess_text(t) for t in resume_texts]

    job_tokens = preprocess_text(job_desc_text)
    job_keywords = _extract_job_keywords(job_tokens)

    use_sbert = (engine or "auto") != "tfidf"
    sbert_sims = _sbert_similarity(job_desc_text, resume_texts) if use_sbert else []
    if sbert_sims:
        cos_scores = np.array(sbert_sims)
    else:
        corpus_tokens = [job_tokens] + resume_tokens_list
        tfidf, _ = _vectorize_corpus(corpus_tokens)
        job_vec = tfidf[0]
        resume_vecs = tfidf[1:]
        cos_scores = cosine_similarity(resume_vecs, job_vec).reshape(-1)

    results = []
    for i, tokens in enumerate(resume_tokens_list):
        kw_score, matched = _keyword_match_score(tokens, job_keywords)
        base_score = float(cos_scores[i])
        if weights:
            ss, es, ds = _component_scores(resume_texts[i], tokens, job_desc_text)
            comp = weights.get("skills", 0.0) * ss + weights.get("experience", 0.0) * es + weights.get("education", 0.0) * ds
            final_score = 0.4 * base_score + 0.6 * comp
        else:
            ss = es = ds = 0.0
            final_score = 0.7 * base_score + 0.3 * kw_score
        improvements = _generate_improvements(tokens, job_keywords, base_score, kw_score, matched)
        yrs, title, edu = extract_experience_education(resume_texts[i])
        sk = extract_skills(resume_texts[i])
        cand_missing = [kw for kw in job_keywords if kw not in set(tokens)]
        results.append({
            "filename": filenames[i],
            "orig_index": i,
            "base_score": round(base_score, 4),
            "keyword_score": round(kw_score, 4),
            "final_score": round(final_score, 4),
            "matched_keywords": matched,
            "missing_keywords": cand_missing,
            "improvements": improvements,
            "years_experience": yrs,
            "last_job_title": title,
            "education_level": edu,
            "hard_skills": sk["hard_skills"],
            "soft_skills": sk["soft_skills"],
            "skill_score": round(ss, 4),
            "experience_score": round(es, 4),
            "education_score": round(ds, 4),
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    skill_counter = Counter()
    for r in results:
        for s in r["hard_skills"]:
            skill_counter[s] += 1
    keyword_summary = {"job_keywords": job_keywords, "total_resumes": len(results), "top_skills": [k for k, _ in skill_counter.most_common(10)]}
    return results, keyword_summary


def rank_resumes_for_roles(resume_files, roles: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for role in roles:
        title = role.get("title") or "Role"
        desc = role.get("desc") or ""
        weights = role.get("weights") or None
        engine = role.get("engine") or None
        results, summary = rank_resumes(resume_files, desc, weights, engine)
        out[title] = {"results": results, "summary": summary, "job_desc": desc, "engine": (engine or "auto")}
    return out