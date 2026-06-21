# AI Hiring Copilot: Two-Stage Hybrid Ranking Pipeline

## 📌 Overview
Keyword-based applicant tracking systems are fundamentally flawed. They miss highly qualified talent due to phrasing differences and fail to account for a candidate's intent or behavioral signals. 

This project solves that by implementing a **Two-Stage Hybrid AI Ranking System** designed to evaluate candidates exactly like a human recruiter would—by understanding context, career trajectory, and platform activity.

## 🏗 Architecture
Our pipeline processes 100,000+ complex JSON candidate profiles and ranks them against a provided Job Description using a two-step approach:

1. **Stage 1: Semantic Shortlisting (Broad Filter)**
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Action: Converts the job description and candidate profiles into vector embeddings. Calculates cosine similarity to rapidly filter 100,000 candidates down to a highly relevant top 100.
2. **Stage 2: Deep Contextual Re-Ranking (The Deep Read)**
   - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Action: Performs a deep, side-by-side textual evaluation of the top 100 candidates against the job description to generate a final fit score.

## 🧠 The "Secret Sauce": Behavioral Signal Integration
Traditional AI matching only looks at "Skills" and "Experience." Our data preprocessing pipeline flattens complex JSON objects to actively integrate **RedRob Behavioral Signals**. By feeding platform activity and intent metrics into the Cross-Encoder, the AI weighs *candidate intent* alongside *technical capability*.

## 🚀 Setup & Execution
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Kaggle notebook / Python script against the raw dataset.
*(Note: Raw candidate datasets are excluded from this repository for privacy and storage efficiency).*
