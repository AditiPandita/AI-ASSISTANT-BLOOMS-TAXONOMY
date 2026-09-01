from fastapi import FastAPI

app = FastAPI(
    title="AI Assistant",
    version="1.0.0"
)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "AI Assistant backend is running"
    }