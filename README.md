# AI Hiring Copilot: Two-Stage Hybrid Ranking Pipeline

## Overview

Keyword-based applicant tracking systems often miss highly qualified candidates because they rely on exact phrase matches rather than understanding role fit, career trajectory, behavioral signals, and intent.

This project solves that problem with a **two-stage hybrid AI ranking system** that evaluates candidates like a strong recruiter would: by reading the job description, understanding the role, comparing the full candidate profile, and producing a trustworthy ranked shortlist.

## Core Idea

The pipeline combines:

1. **Fast semantic filtering** to narrow down a large candidate pool.
2. **Deep contextual re-ranking** to identify the best matches from the shortlist.

It is designed to work with:
- structured candidate JSONL data,
- job description documents,
- schema files,
- sample submissions,
- and validation scripts provided by the challenge.

## Architecture

### High-Level Flow

```text
Input: Job Description + Candidate Profiles
        ↓
Parse and clean job description
        ↓
Flatten and normalize candidate records
        ↓
Generate semantic embeddings
        ↓
Compute cosine similarity for broad filtering
        ↓
Cross-encoder re-ranking on top candidates
        ↓
Blend semantic + contextual + behavioral signals
        ↓
Generate final ranked shortlist
        ↓
Output: Submission file + explanation artifacts
```

### Stage 1: Semantic Shortlisting

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Purpose:** Convert the job description and each candidate profile into embeddings.
- **Method:** Use cosine similarity to quickly identify the most relevant candidates.
- **Benefit:** Scales well to large pools and removes obvious mismatches early.

### Stage 2: Deep Contextual Re-Ranking

- **Model:** `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Purpose:** Evaluate the job description and candidate profile together as a pair.
- **Method:** Score the top semantic matches with a richer contextual model.
- **Benefit:** Produces better ranking quality than embeddings alone.

## Behavioral Signal Integration

Traditional matching systems focus too heavily on skills and keywords. This solution also incorporates behavioral and platform signals from the challenge data.

Examples of signals that can improve ranking:
- activity and engagement,
- consistency of career progression,
- completeness of profile,
- evidence of role alignment,
- project and experience relevance,
- profile freshness and intent markers.

These signals help the model reflect real recruiter reasoning rather than just text overlap.

## Data Inputs

The challenge package includes files such as:

- `candidates.jsonl`
- `candidate_schema.json`
- `job_description.docx`
- `redrob_signals_doc.docx`
- `sample_submission.csv`
- `validate_submission.py`
- `submission_metadata_template.yaml`
- `submission_spec.docx`
- `README.docx`
- `sample_candidates.json`

## Repository Structure

```text
.
AI-Hiring-Copilot/
│
├── README.md
├── notebook76a10792d5.ipynb
├── requirements.txt
└── validate_submission.py
```

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Recommended Libraries

- `pandas`
- `numpy`
- `scikit-learn`
- `sentence-transformers`
- `python-docx`
- `pyyaml`
- `openpyxl`

## How to Run

### Kaggle Notebook Workflow

1. Add the challenge ZIP or extracted dataset as input.
2. Locate the innermost challenge folder.
3. Load the candidate JSONL file.
4. Read the job description DOCX.
5. Inspect the schema and sample submission.
6. Generate candidate embeddings.
7. Rank candidates in two stages.
8. Save the final submission file.
9. Validate the output with the provided script.

### Example Usage

```bash
python solution.py   --jd_path "job_description.docx"   --candidates_path "candidates.jsonl"   --output_path "submission.csv"
```

## Implementation Details

### 1. Document Parsing
Extract text from the job description DOCX and normalize it for downstream scoring.

### 2. Candidate Profile Processing
Load the JSONL candidate records and flatten nested fields into a unified textual representation.

### 3. Embedding Generation
Use a MiniLM embedding model to represent both the job description and candidate profiles.

### 4. Semantic Ranking
Compute cosine similarity and keep the strongest candidates for deeper analysis.

### 5. Cross-Encoder Ranking
Score the top candidates with a pairwise relevance model to capture contextual fit.

### 6. Signal Fusion
Combine semantic similarity, contextual score, and behavioral features into a final fit score.

## Scoring Strategy

A practical final score can be written as:

```text
Final Score = α × Semantic Similarity + β × Cross-Encoder Score + γ × Behavioral Signal Score
```

Where:
- `α` controls broad relevance,
- `β` controls contextual fit,
- `γ` controls behavioral quality,
- and the weights can be tuned empirically.

## Output Format

The final output should match the sample submission exactly.

Typical fields may include:
- candidate ID
- rank
- score
- supporting reason or metadata if required

Always validate the file against the provided validator before submitting.

## Validation

Use the supplied validation script to check:
- file format,
- required columns,
- row order,
- ranking constraints,
- and metadata compatibility.

## Why This Approach Works

- **Better recall than keyword matching**
- **Better precision than embeddings alone**
- **More recruiter-like decision making**
- **Scales to large candidate pools**
- **Produces explainable shortlist decisions**

## Testing Checklist

Before submission, confirm that:

- the job description loads without errors,
- candidate records parse correctly,
- embeddings are generated successfully,
- the ranking pipeline produces sorted results,
- the submission file matches the sample format,
- the validator passes.

## Future Improvements

Possible enhancements include:

- fine-tuning on hiring-domain data,
- learning optimal ranking weights,
- extracting skills and roles more accurately,
- adding explainability summaries,
- bias and fairness audits,
- recruiter feedback loops,
- a dashboard for reviewing shortlisted candidates.

## Responsible AI and Ethics

This project should be used as a decision support system, not as an automatic hiring decision-maker. Ranking should remain transparent, auditable, and subject to human review.

Key principles:
- avoid using protected attributes,
- maintain traceability of scores,
- keep human oversight in the loop,
- monitor for bias and drift,
- explain ranking rationale clearly.

## Conclusion

The **AI Hiring Copilot** combines fast semantic retrieval with contextual re-ranking and behavioral signal fusion to produce a recruiter-quality shortlist. It is designed to be practical, scalable, and explainable, while staying aligned with the challenge objective of building a smarter hiring system.

## Team

**SMASHERS**

## Challenge

**India.Runs**
