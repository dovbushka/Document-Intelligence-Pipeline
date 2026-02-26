# Document Intelligence Pipeline

This project is a production-style OCR + LLM pipeline designed to extract structured data from scanned documents at scale.

The idea behind this system was simple:  
Can we build something that processes thousands of documents per day, extracts structured fields reliably, and tells us how confident it is about the result?

This repository is my implementation of that system.

---

## What This Does

- Accepts scanned documents (images)
- Runs OCR (Tesseract) to extract raw text
- Uses GPT-4 to extract structured fields
- Runs multiple extraction passes (ensemble approach)
- Calculates a confidence score based on agreement
- Returns validated JSON output

The goal is reliability and scalability, not just “LLM magic”.

---

## Tech Stack

- Python
- FastAPI
- Pydantic
- OpenAI (GPT-4)
- Tesseract OCR
- Docker-ready

---

## How It Works (High Level)

1. A document is uploaded through the API.
2. OCR extracts the raw text.
3. The text is sent to GPT for structured extraction.
4. The extraction runs multiple times.
5. Results are compared (ensemble voting).
6. A confidence score is calculated.
7. Clean JSON is returned.

---

## Why Ensemble?

LLMs can vary slightly between runs.  
Instead of trusting a single output, the system:

- Runs extraction multiple times
- Compares outputs
- Selects the most common result
- Computes agreement ratio as confidence

This improves stability when processing large volumes.

---

## Running Locally

### 1. Install system dependency

If you're using Ubuntu / Codespaces:

```bash
sudo apt update
sudo apt install -y tesseract-ocr
