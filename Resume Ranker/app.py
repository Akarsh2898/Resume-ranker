import io
import os
import uuid
import time
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from resume_ranker.processor import rank_resumes, generate_csv_report, rank_texts, rank_resumes_for_roles, extract_text_from_pdf_path, generate_combined_csv_report
from resume_ranker.samples import SAMPLE_JOB_DESC, SAMPLE_RESUMES


app = Flask(__name__)
app.secret_key = "resume-ranker-secret"


@app.route("/", methods=["GET"]) 
def index():
    return render_template("index.html")


@app.route("/rank", methods=["POST"]) 
def rank():
    files = request.files.getlist("resumes")

    resume_files = [f for f in files if f and f.filename.lower().endswith(".pdf")]

    role_titles = request.form.getlist("role_title[]")
    role_descs = request.form.getlist("role_desc[]")
    weight_skills = request.form.getlist("weight_skills[]")
    weight_experience = request.form.getlist("weight_experience[]")
    weight_education = request.form.getlist("weight_education[]")
    role_engine = request.form.getlist("role_engine[]")
    multi_roles = []
    for i in range(len(role_descs)):
        t = role_titles[i] if i < len(role_titles) else "Role"
        d = role_descs[i].strip()
        if not d:
            continue
        try:
            ws = float(weight_skills[i]) if i < len(weight_skills) else 0.4
            we = float(weight_experience[i]) if i < len(weight_experience) else 0.3
            wd = float(weight_education[i]) if i < len(weight_education) else 0.3
        except Exception:
            ws, we, wd = 0.4, 0.3, 0.3
        eng = role_engine[i] if i < len(role_engine) else "auto"
        multi_roles.append({"title": t or "Role", "desc": d, "weights": {"skills": ws, "experience": we, "education": wd}, "engine": eng})

    single_job_desc = request.form.get("job_desc", "").strip()
    job_title = request.form.get("job_title", "").strip() or "Job Description"
    ws_single = request.form.get("weight_skills_single", "0.4")
    we_single = request.form.get("weight_experience_single", "0.3")
    wd_single = request.form.get("weight_education_single", "0.3")
    engine_single = request.form.get("engine_single", "auto").strip()
    try:
        single_weights = {"skills": float(ws_single), "experience": float(we_single), "education": float(wd_single)}
    except Exception:
        single_weights = {"skills": 0.4, "experience": 0.3, "education": 0.3}
    if not multi_roles and not single_job_desc:
        flash("Please provide at least one job description.")
        return redirect(url_for("index"))

    if not resume_files:
        flash("Please upload at least one PDF resume.")
        return redirect(url_for("index"))

    try:
        upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        stored = []
        resume_text_items = []
        for f in resume_files:
            fid = str(uuid.uuid4())
            safe_name = f.filename
            path = os.path.join(upload_dir, f"{fid}_{safe_name}")
            f.save(path)
            text = extract_text_from_pdf_path(path)
            stored.append({"id": fid, "filename": safe_name, "path": path})
            resume_text_items.append((safe_name, text))

        if multi_roles:
            multi = {}
            for role in multi_roles:
                title = role.get("title")
                desc = role.get("desc")
                weights = role.get("weights")
                res, summ = rank_texts(list(resume_text_items), desc, weights, role.get("engine"))
                for r in res:
                    idx = r.get("orig_index", 0)
                    r["file_id"] = stored[idx]["id"]
                multi[title] = {"results": res, "summary": summ, "job_desc": desc, "engine": role.get("engine"), "weights": weights}
        else:
            results, keyword_summary = rank_texts(list(resume_text_items), single_job_desc, single_weights, engine_single)
            for r in results:
                idx = r.get("orig_index", 0)
                r["file_id"] = stored[idx]["id"]
        try:
            now = time.time()
            for fn in os.listdir(upload_dir):
                p = os.path.join(upload_dir, fn)
                if os.path.isfile(p) and now - os.path.getmtime(p) > 6 * 3600:
                    os.remove(p)
        except Exception:
            pass
    except Exception as e:
        flash(f"Error processing resumes: {e}")
        return redirect(url_for("index"))

    # Persist results in session for download
    if multi_roles:
        session["multi"] = multi
        session["uploads_map"] = stored
        return render_template(
            "results.html",
            multi=multi,
            job_title="Multiple Roles",
        )
    else:
        session["last_results"] = results
        session["last_job_title"] = job_title
        session["last_keyword_summary"] = keyword_summary
        session["uploads_map"] = stored
        return render_template(
            "results.html",
            job_title=job_title,
            job_desc=single_job_desc,
            results=results,
            keyword_summary=keyword_summary,
            engine=engine_single,
            weights=single_weights,
        )


@app.route("/rank_sample", methods=["GET"]) 
def rank_sample():
    job_title = "Sample: Machine Learning Engineer"
    results, keyword_summary = rank_texts(SAMPLE_RESUMES, SAMPLE_JOB_DESC)
    session["last_results"] = results
    session["last_job_title"] = job_title
    session["last_keyword_summary"] = keyword_summary
    return render_template(
        "results.html",
        job_title=job_title,
        job_desc=SAMPLE_JOB_DESC,
        results=results,
        keyword_summary=keyword_summary,
    )


@app.route("/download_report", methods=["GET"]) 
def download_report():
    last_results = session.get("last_results")
    job_title = session.get("last_job_title", "Job Description")
    if not last_results:
        flash("No report available. Please run a ranking first.")
        return redirect(url_for("index"))

    csv_bytes, filename = generate_csv_report(last_results, job_title)
    return send_file(
        io.BytesIO(csv_bytes),
        as_attachment=True,
        download_name=filename,
        mimetype="text/csv",
    )


@app.route("/download_report_role/<role_title>", methods=["GET"]) 
def download_report_role(role_title):
    multi = session.get("multi")
    if not multi or role_title not in multi:
        flash("No report available for this role.")
        return redirect(url_for("index"))
    payload = multi[role_title]
    results = payload.get("results")
    csv_bytes, filename = generate_csv_report(results, role_title)
    return send_file(
        io.BytesIO(csv_bytes),
        as_attachment=True,
        download_name=filename,
        mimetype="text/csv",
    )


@app.route("/download_report_all", methods=["GET"]) 
def download_report_all():
    multi = session.get("multi")
    if not multi:
        flash("No multi-role report available.")
        return redirect(url_for("index"))
    csv_bytes, filename = generate_combined_csv_report(multi)
    return send_file(
        io.BytesIO(csv_bytes),
        as_attachment=True,
        download_name=filename,
        mimetype="text/csv",
    )


if __name__ == "__main__":
    # Run Flask dev server
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)

@app.route("/view_resume/<file_id>", methods=["GET"])
def view_resume(file_id):
    uploads = session.get("uploads_map", [])
    match = next((u for u in uploads if u["id"] == file_id), None)
    if not match:
        flash("File not found.")
        return redirect(url_for("index"))
    return send_file(match["path"], as_attachment=False)

@app.route("/download_resume/<file_id>", methods=["GET"])
def download_resume(file_id):
    uploads = session.get("uploads_map", [])
    match = next((u for u in uploads if u["id"] == file_id), None)
    if not match:
        flash("File not found.")
        return redirect(url_for("index"))
    return send_file(match["path"], as_attachment=True, download_name=match["filename"])