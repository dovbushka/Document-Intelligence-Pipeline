import os


structure = {
    "app/__init__.py": "",
    "app/main.py": """from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Document Intelligence Pipeline")

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "running"}
""",

    "app/config.py": """import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MODEL_NAME: str = "gpt-4o"

settings = Settings()
""",

    "app/api/__init__.py": "",
    "app/api/routes.py": """from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.core.pipeline import DocumentPipeline

router = APIRouter()

@router.post("/process")
async def process_document(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = DocumentPipeline.process(file_path)

    os.remove(file_path)
    return result
""",

    "app/models/__init__.py": "",
    "app/models/schemas.py": """from pydantic import BaseModel, Field
from typing import Dict, Any

class ExtractedDocument(BaseModel):
    fields: Dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0)
""",

    "app/services/__init__.py": "",
    "app/services/ocr_service.py": """import pytesseract
from PIL import Image

class OCRService:
    @staticmethod
    def extract_text(image_path: str) -> str:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
""",

    "app/services/vision_llm_service.py": """from openai import OpenAI
from app.config import settings
import json

client = OpenAI(api_key=settings.OPENAI_API_KEY)

class VisionLLMService:
    @staticmethod
    def extract_structured_data(text: str) -> dict:
        prompt = f\"\"\"
        Extract structured form fields from the following document text.
        Return valid JSON only.

        TEXT:
        {text}
        \"\"\"

        response = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a document extraction AI."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )

        return json.loads(response.choices[0].message.content)
""",

    "app/core/__init__.py": "",
    "app/core/confidence.py": """def calculate_confidence(outputs: list) -> float:
    if not outputs:
        return 0.0

    agreement = max(outputs.count(o) for o in outputs)
    return round(agreement / len(outputs), 2)
""",

    "app/core/ensemble.py": """from collections import Counter

def ensemble_vote(outputs: list) -> dict:
    counter = Counter(str(o) for o in outputs)
    best = counter.most_common(1)[0][0]
    return eval(best)
""",

    "app/core/pipeline.py": """from app.services.ocr_service import OCRService
from app.services.vision_llm_service import VisionLLMService
from app.core.ensemble import ensemble_vote
from app.core.confidence import calculate_confidence

class DocumentPipeline:

    @staticmethod
    def process(image_path: str):
        text = OCRService.extract_text(image_path)

        outputs = []
        for _ in range(3):  # multi-model simulation
            result = VisionLLMService.extract_structured_data(text)
            outputs.append(result)

        voted = ensemble_vote(outputs)
        confidence = calculate_confidence(outputs)

        return {
            "fields": voted,
            "confidence_score": confidence
        }
""",

    "requirements.txt": """fastapi
uvicorn
pydantic
python-multipart
openai
pillow
pytesseract
python-dotenv
httpx
""",

    ".env": """OPENAI_API_KEY=your_api_key_here
""",

    "Dockerfile": """FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
}

for path, content in structure.items():
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Project created successfully!")
