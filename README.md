# 🎯 From Headline to Hiring Signal

### Teaching a 3B-parameter language model to score candidates the way a recruiter does, and to say so when it isn't sure

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Transformers-PEFT-FFD21E?logo=huggingface&logoColor=black)
![Model](https://img.shields.io/badge/Qwen2.5--3B-QLoRA%20(NF4)-6E40C9)
![Retrieval](https://img.shields.io/badge/RAG-FAISS%20%7C%20ChromaDB-0A9396)
![Kaggle](https://img.shields.io/badge/Built%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)

---

## 🏢 The problem: finding talent is still a manual craft

This project started with a real business problem. **[Apziva](https://www.apziva.com)** brought in a client, a **talent sourcing and management company** whose job is to find talented people and place them with technology companies, and I developed the solution together with Apziva as an **AI Resident**.

On paper, the work sounds simple: a client has a role, and the firm finds the right person for it. In practice, every placement depends on three hard questions:

1. **What does the role really need?** Filling a position well means deeply understanding the client: what they're looking for, which skills matter, and what "good" looks like for them.
2. **What makes a candidate shine for *this* role?** A strong data engineer and a strong HR coordinator stand out for completely different reasons. Recognizing fit takes experience.
3. **Where are the right people?** Talented individuals are scattered and hard to find.

Today, all of this runs on human effort. Recruiters search with keywords such as *"full-stack software engineer"*, *"engineering manager"* or *"aspiring human resources"*, and the keywords change with every new role. They then go through the resulting candidates one by one, and every profile has to be reviewed by hand to judge how good a fit it really is.

The review step brings its own twist. After a careful look, the best candidate is often **not** the one at the top of the list. It might be the 7th. The firm wants to capture that signal: when a reviewer **stars** a candidate as the ideal fit for a role, the list should **re-rank itself** around that choice, so every review makes the next list smarter.

Sourcing itself was already semi-automated, so the brief was clear about where to focus:

> **Build a machine-learning pipeline that understands candidates, scores how well they fit, ranks them, and learns from the reviewers who use it.**

---

## 🤝 Built in collaboration with Apziva

This was not a classroom exercise. **Apziva** secured the project from the client and framed the business problem, and I worked together with the Apziva team to develop the solution through its **AI Residency** program.

The roles were clear:

| | Contribution |
|---|---|
| **The client** | A talent sourcing and management company that provided the business need, the candidate data, and the screening scores used as training labels |
| **Apziva** | Brought in the client, defined the project scope and requirements, and collaborated throughout development |
| **Me (AI Resident)** | Designed and built the pipeline end to end: LLM-based extraction, QLoRA fine-tuning of the scoring model, the FAISS and ChromaDB retrieval layers, evaluation, and this write-up |

Working on a real client problem shaped every decision in this repository. The goal was never just a good metric on a leaderboard. It was a system a recruiting team could actually trust, question, and use.

---

## 📖 How I approached it

A recruiter can glance at a one-line profile ("ms in data analytics, northeastern university, open to data roles") and form a judgment in seconds. That judgment is the firm's most valuable asset, but it lives in people's heads. It isn't written down anywhere, it doesn't scale to thousands of applicants, and nobody can audit it.

So I set out to capture it. I built the system in **three stages**, each in its own notebook, and each answering a different part of the brief:

- First, the machine learns to **read**: messy candidate text becomes clean, structured JSON.
- Then it learns to **judge**: a language model is fine-tuned to predict the screening score the firm's reviewers assigned.
- Finally, it learns to **know when it's out of its depth**: retrieval over past candidates calibrates each score, explains it with real examples, and flags the cases a human should look at more closely.

### From the brief to the build

| What the firm needed | How this project answers it | Where |
|---|---|---|
| Understand candidates whose profiles are written in a hundred different ways | A few-shot LLM turns raw text into consistent fields; job titles are normalized so the same role always has the same name, which makes keyword matching reliable | Chapter 1 |
| Recognize what makes a candidate a good fit | The model learns directly from the firm's own screening scores, absorbing the reviewers' judgment instead of hand-written rules | Chapter 2 |
| Rank candidates by fitness | Every candidate gets a 0–100 fitness score; ranking quality is measured with Spearman correlation against the reviewers' order | Chapter 2 |
| Reduce the cost of manual review | Each score comes with the most similar past candidates as evidence, plus a low-confidence flag that points reviewers to the cases that need them most | Chapter 3 |
| Re-rank when a reviewer stars the ideal candidate | The retrieval layer's embedding space is the foundation for this feedback loop; it is the next stage on the roadmap | [Roadmap](#-roadmap-closing-the-loop-with-starring) |

---

## 🗺️ The pipeline at a glance

| # | Notebook | What it does |
|---|---|---|
| 1 | [`extraction-candidate-info_final.ipynb`](notebooks/extraction-candidate-info_final.ipynb) | Turns raw candidate text into structured JSON with a few-shot LLM |
| 2 | [`fine-tuning-candidates_final.ipynb`](notebooks/fine-tuning-candidates_final.ipynb) | QLoRA fine-tunes Qwen2.5-3B to predict the screening score |
| 3a | [`03a_rag-faiss-system.ipynb`](notebooks/03a_rag-faiss-system.ipynb) | Retrieval layer on a FAISS index: blend, explain, flag |
| 3b | [`03b_rag-chroma-system.ipynb`](notebooks/03b_rag-chroma-system.ipynb) | Same retrieval layer on ChromaDB, plus metadata filtering |

---

## 🔍 Chapter 1: Teaching the machine to read

**Notebook:** `extraction-candidate-info_final.ipynb`

Real candidate data is messy. One person writes *"HR Manager with 5 years experience"*. Another writes a run-on string of skills with no punctuation. A third writes *"Passionate about helping people"* and nothing else. Before any model can score these people, it needs them in a consistent shape. This is the first challenge from the brief in miniature: you can't judge fit until you understand who the candidate is.

So the first thing I did was hand the problem to **Qwen2.5-3B-Instruct**, running locally, with a carefully written few-shot prompt that extracts four fields from every candidate:

```json
{
  "job_title":  "data analyst",
  "experience": "student/intern-level",
  "education":  "ms in data analytics engineering, northeastern university, boston",
  "other_info": null
}
```

The prompt does more than copy text. It **normalizes abbreviations** so equivalent roles collapse to one string ("HR" → "human resources", "SWE" → "software engineer"). This matters for a keyword-driven search: a search for "human resources" should find the people who wrote "HR". When a student only names their field of study, it **infers the target role** they're pursuing, which is exactly how an "aspiring human resources" candidate shows up. And it is explicitly told **never to invent** information that isn't in the text. Every value is lowercased so "Data Analyst" and "data analyst" can never become two different categories.

**Making it fast and robust:**

- **Batched generation.** 1,266 one-at-a-time generations would leave the GPU idle between calls, so I process 25 candidates per `generate()` call. Decoder-only models need **left padding** for this to work, so every sequence in a batch ends at the same position and generation continues from one shared point.
- **Per-row retry.** If one candidate in a batch produces malformed JSON, I don't throw the batch away. Only that row is regenerated on its own.
- **Self-auditing.** After extraction, the notebook checks itself: blank rates per field, casing inconsistencies across titles, and whether any generation hit the token ceiling and got truncated.

**Results**

| Metric | Value |
|---|---|
| Candidates extracted | **1,266** |
| JSON parse errors | **0** |
| Distinct job titles after normalization | 260 (no casing duplicates) |
| `job_title` filled | 94.1% |
| `experience` / `education` / `other_info` filled | 59.9% / 43.8% / 50.7% |
| Average tokens generated per candidate | 62.5 |

The blank rates for experience and education are expected rather than a failure: most one-line profiles simply don't mention them, and the prompt is designed to return `null` instead of guessing.

---

## 🧠 Chapter 2: Teaching the machine to judge

**Notebook:** `fine-tuning-candidates_final.ipynb`

Now each candidate is a clean JSON record with a reviewer-assigned **screening score from 0 to 100**. This is the second challenge from the brief, knowing what makes a candidate shine, turned into a learning problem: *can a model learn the reviewers' judgment from the profile alone?*

### Turning JSON into model input

Each record becomes one line of text the model can read:

```
Job Title: data analyst | Experience: 3 years | Education: b.sc. statistics | Other Info: sql, power bi, python | Location: us
```

The screening score is **never** part of this text. That is the real guarantee against leakage, and an explicit check confirms it afterwards. The check uses word-boundary matching rather than a naive substring search, which would raise false alarms when a score of 25 appears inside "2025".

### A split that's fair to every job title

The candidates cover 260 different titles, very unevenly. A random split could easily put most data analysts in training and most data scientists in testing, purely by chance. So I used a **stratified 70/15/15 split by primary job title**, merging titles with fewer than 10 candidates into an "Other" bucket for stratification only. I also verified that **no candidate appears in more than one split**.

| Train | Validation | Test |
|---|---|---|
| 886 | 190 | 190 |

### Why QLoRA

Fully fine-tuning a 3-billion-parameter model needs far more GPU memory than Kaggle offers. QLoRA makes it possible:

- The base model is **loaded in 4-bit NF4** (a quantization scheme designed for the bell-shaped distribution of neural network weights) with double quantization, and stays **frozen**.
- Small **LoRA adapters** (rank 16, alpha 32) are attached to every attention and MLP projection: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Qwen's text-generation head is swapped for a **single-output regression head** (`score`). Because that head starts out random, it's kept **fully trainable** via `modules_to_save`, since a low-rank nudge alone wouldn't be enough to train it.

The result: only **29.9M of 3.1B parameters (0.96%)** are trained.

### Training on the metric that matters

The default regression loss is MSE, which punishes big errors disproportionately. I wrote a small custom `MAETrainer` that trains directly on **L1 loss (MAE)**, so the number the model optimizes is the same number I evaluate it on. Training ran with an effective batch size of 16, learning rate 2e-4 with 5% warmup, bf16 mixed precision, gradient checkpointing, and early stopping. The best checkpoint came from **epoch 8 of 10** and was restored automatically.

### Results on the untouched test set

The test set was never used for model selection, tuning, or early stopping. It was evaluated exactly once.

| Model | Test MAE ↓ | Test RMSE ↓ | Spearman ρ ↑ |
|---|---|---|---|
| Naive baseline (always predict the training mean) | 30.25 | n/a | n/a |
| TF-IDF + Ridge *(validation set, for reference)* | 22.91 | n/a | 0.322 |
| **QLoRA Qwen2.5-3B (this project)** | **16.57** | **25.30** | **0.548** |

The fine-tuned model **cuts the naive baseline's error by about 45%** and ranks candidates in substantially the same order the reviewers did. For a firm that works from ranked shortlists, that ranking agreement is the number that matters most.

### What the model actually learned

I didn't want to stop at a number, so the notebook takes the model apart afterwards.

**The labels are two-tiered.** Reviewer scores cluster at 20–40 and 80–100, with **not one training candidate between 40 and 60**. Reviewers weren't grading on a smooth scale; they were effectively sorting candidates into "strong" and "weak". This explains the error profile: when the model picks the wrong tier, the miss is large. The worst 10% of test predictions account for **33.5% of all error**.

**Some disagreement is in the labels themselves.** Twenty identical input texts appear more than once in training with different scores. No model can beat that noise floor.

**Which fields drive the score?** Using permutation importance on the validation set (shuffle one field, measure how much MAE gets worse):

| Field | MAE increase when shuffled |
|---|---|
| `location` | **+7.32** |
| `job_title` | +5.26 |
| `experience` | +1.06 |
| `education` | +0.80 |
| `other_info` | +0.71 |

Location matters more than anything else, which is an important finding in itself. I come back to it in [Responsible use](#%EF%B8%8F-responsible-use--limitations) below.

---

## 📚 Chapter 3: Teaching the machine to know what it doesn't know

**Notebooks:** `03a_rag-faiss-system.ipynb` and `03b_rag-chroma-system.ipynb`

A score on its own isn't something a recruiter can act on. They want to know *why* a candidate got 81, and *whether the model has seen anyone like this before*. Remember that every candidate in this firm's process is still reviewed by a person. The goal of this stage is to make that review faster and better aimed, not to replace it.

### What "RAG" means for a regression model

My fine-tuned model outputs a number, not text, so retrieval here doesn't mean "fetch context and let an LLM write an answer". It does three other jobs:

1. **Calibration.** Find the 5 most similar past candidates and blend the model's score with their similarity-weighted average score.
2. **Out-of-distribution detection.** If even the closest past candidate isn't very similar, flag the prediction as low-confidence, because the model is extrapolating.
3. **Explainability.** Show the recruiter the actual past candidates the new one resembles, so the score is grounded in examples they can check.

### One forward pass, two outputs

Rather than bringing in a separate embedding model, I reuse the fine-tuned model itself. Qwen's regression head reads the hidden state of the **last real token**, so I pull that same 2,048-dimensional vector from the same forward pass that produces the score. The embedding comes for free, and it lives in exactly the space the model uses to make its judgment, so "similar" means *similar in the ways that matter for scoring*.

### Why two vector stores?

I built the retrieval layer twice on purpose, to compare the trade-offs:

| | FAISS (`03a`) | ChromaDB (`03b`) |
|---|---|---|
| Index | `IndexFlatIP` on L2-normalized vectors (exact cosine) | HNSW with cosine space |
| Metadata | Separate CSV, aligned by row position | Stored alongside each vector |
| Persistence | Manual `write_index` + CSV | Automatic via `PersistentClient` |
| Filtering | Not built in | `where=` filters, e.g. only retrieve neighbours with the same job title or location |
| Best for | Raw speed, minimal dependencies | Production-style workflows with rich metadata |

ChromaDB's metadata filtering maps naturally onto the firm's keyword-driven workflow: a search for "engineering manager" can restrict retrieval to neighbours with that same normalized title.

Both indexes are built from the **training set only**. Validation and test candidates are only ever queries, never index members, so a candidate can never retrieve itself.

### Tuning the blend

The blend weight `alpha` (1.0 = pure model, 0.0 = pure neighbours) was grid-searched on the **validation set**. Both stores independently chose **alpha = 0.8**, and the result was then locked and evaluated on test once.

| Approach | Test MAE ↓ | Test RMSE ↓ | Spearman ρ ↑ |
|---|---|---|---|
| Regression only | 15.95 | 24.67 | 0.568 |
| RAG-blended (FAISS) | 16.03 | 23.88 | **0.577** |
| RAG-blended (ChromaDB) | 16.04 | **23.87** | 0.576 |

*The regression-only numbers here differ slightly from Chapter 2 because the retrieval notebooks reload the adapter with float16 compute instead of bfloat16. Comparisons within this table are like-for-like.*

The honest reading: blending **leaves MAE essentially flat but trims RMSE by about 3%** and slightly improves ranking. Neighbours pull back some of the large wrong-tier misses, which is exactly where the two-tier labels hurt most. The real value of this layer is less about the headline number and more about the **confidence flag and the explanations**. Only 1 of 190 test candidates was flagged as out-of-distribution, so the model is rarely extrapolating on this data.

---

## 🆕 Scoring a brand-new candidate

This is the whole system working end to end. A new candidate arrives as JSON, in the same schema the extraction stage produces:

```python
new_candidate = {
    "job_title":  "Data Analyst",
    "experience": "3 years",
    "education":  "B.Sc. Statistics",
    "other_info": "SQL, Power BI, Python",
    "location":   "US",
}

result = score_with_rag(build_input_text(new_candidate), alpha=0.8, k=5)
```

What comes back:

```python
{
    "regression_score": ...,    # the fine-tuned model's own prediction
    "knn_score":        ...,    # similarity-weighted average of 5 nearest past candidates
    "final_score":      81.0,   # 0.8 × regression + 0.2 × kNN
    "low_confidence":   False,  # closest neighbour is similar enough to trust
    "top_similarity":   ...,
    "neighbors":        [...]   # the 5 past candidates it resembles, with their real scores
}
```

For a reviewer, that's a score, a reason, and a confidence level in one result.

The notebooks also include a **what-if probe**: remove or change one field at a time and watch the score move. For this candidate, the full profile scores 79.5 from the regression head. Removing education or other skills barely changes it, but **removing location drops it to 62.5**. That's the feature-importance finding from Chapter 2, visible on a single person.

---

## 🔭 Roadmap: closing the loop with starring

The brief's most interesting requirement is still ahead: **when a reviewer stars the ideal candidate, the list should re-rank itself.** This repository doesn't implement starring yet, but the retrieval layer was designed with it in mind.

Because every candidate already lives in the model's own judgment space, a starred candidate can become an **anchor**. Re-ranking then means blending each candidate's model score with how close they sit to the starred ideal in that space, the same blending mechanism Chapter 3 already uses with past candidates. Each new star adds an anchor, so the ranking moves closer to what the reviewer actually wants with every review.

Next steps on this path:

- Store starred candidates per role, alongside the role's search keywords.
- Re-rank the shortlist by combining model score and similarity to the starred anchors, with the blend weight tuned on historical review decisions.
- Measure success by how quickly the reviewers' final choices rise toward the top of the list.

---

## 💡 What I learned building this

- **Look at the label distribution before choosing a loss.** The two-tier scores explained more about the model's errors than any hyperparameter did.
- **Qwen is not BERT.** No default pad token, a regression head called `score` instead of `classifier`, last-token pooling instead of `[CLS]`, and different LoRA module names. Each of these fails in its own confusing way if you carry BERT habits over.
- **Precision settings must agree.** The 4-bit compute dtype and the trainer's mixed-precision mode have to match, or you get dtype errors deep inside backprop.
- **Retrieval doesn't have to mean generation.** For a scoring model, the most useful retrieval outputs are calibration, a confidence signal, and evidence a human can check.
- **A cheap ablation before an expensive one.** Before retraining a 3B model to test whether a suspect `experience` tag was hurting, I ran the same experiment with TF-IDF + Ridge in seconds. It said the tag wasn't the problem, which saved a full training run.

---

## ⚖️ Responsible use & limitations

This model learns to reproduce **historical screening judgments**, including whatever patterns those judgments contained. A few things anyone using it should know:

- **Location is the strongest signal.** The model relies on where a candidate is based more than on their role, experience, or education. That's a faithful reflection of the training labels, not a design choice. Location can act as a proxy for characteristics that must not influence hiring decisions, so this is the first thing I'd audit before any real-world use.
- **Small, imbalanced data.** 1,266 candidates across 260 titles, with 240 titles having fewer than 5 examples. Predictions for rare roles rest on very little evidence.
- **Label noise.** Identical profiles received different scores, which caps how accurate any model can be.
- **Decision support, not decision-making.** The score, neighbours, and confidence flag are meant to help a human reviewer prioritize and question, never to reject a candidate automatically. That matches how the firm works: every candidate is still reviewed by a person.

> **Data privacy:** the candidate dataset was provided by the client through Apziva. It, the extracted JSON, and the vector indexes are derived from real people's profiles and are **not** included in this repository. The input schema below describes what the pipeline expects, so you can run it on your own data.

---

## 🚀 Running it yourself

The notebooks were built and run on **Kaggle GPUs**.

1. **Install dependencies**
   ```bash
   pip install -U transformers peft accelerate datasets scikit-learn scipy "bitsandbytes>=0.46.1" torchao
   pip install faiss-cpu chromadb
   ```
2. **Run the notebooks in order**: extraction → fine-tuning → `03a` and/or `03b`. Restart the runtime after installing bitsandbytes.
3. **Supply your data** in the schema below, and update the paths in each notebook's config cell.

**Input schema** (output of notebook 1, input to notebooks 2 and 3):

| Field | Type | Description |
|---|---|---|
| `id` | int | Unique candidate id |
| `raw_text` | str | Original headline / bio text |
| `job_title` | str \| null | Normalized role |
| `experience` | str \| null | Seniority or years, only if stated |
| `education` | str \| null | Degree / institution |
| `other_info` | str \| null | Skills, certifications, employer |
| `location` | str \| null | Candidate location |
| `screening_score` | float | Reviewer-assigned score, 0–100 (training label) |

**Trained adapter:** the LoRA adapter and regression head are published on Hugging Face at
👉 [`DivyeshBhavsar10/qwen2.5-3b-candidate-scorer-qlora`](https://huggingface.co/DivyeshBhavsar10/qwen2.5-3b-candidate-scorer-qlora)

---

## 🛠️ Tech stack

**Models:** Qwen2.5-3B-Instruct (extraction and fine-tuning base), **Built with Qwen**
**Fine-tuning:** Hugging Face Transformers, PEFT (LoRA), bitsandbytes (4-bit NF4 QLoRA)
**Retrieval:** FAISS, ChromaDB
**Evaluation:** scikit-learn, SciPy (MAE, RMSE, Spearman, permutation importance)
**Compute:** Kaggle GPU notebooks

---

## 📄 License

The code in this repository is released under the MIT License.
The fine-tuned adapter on Hugging Face is **Built with Qwen** and is distributed under the [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE).

---

## 👤 About

Built by **Divyesh Bhavsar**, a data scientist working across NLP, ML engineering, and data pipelines, as an AI Resident with **Apziva**.

**Acknowledgements:** thank you to the Apziva team for bringing this real-world project to the residency and for the collaboration throughout development, and to the client for the business problem and data that made it possible.

[LinkedIn](https://www.linkedin.com/in/divyesh-bhavsar-aaaa98152) · [Kaggle](https://www.kaggle.com/divyeshbhavsar) · [Hugging Face](https://huggingface.co/DivyeshBhavsar10)

*If this project was useful or interesting, a ⭐ is always appreciated.*