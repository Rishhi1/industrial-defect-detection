from fastapi import FastAPI

app = FastAPI(
    title="Industrial Defect Detection API",
    description="AI-based industrial visual inspection system",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Industrial Defect Detection API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
