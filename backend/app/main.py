from fastapi import FastAPI, UploadFile, File
from app.ingestion.detector import detect_file_type
from app.ingestion.router import extract_content

app = FastAPI(
    title="AI Assistant",
    version="1.0.0"
)

#To check the health of the backend we can create a simple endpoint that returns a status message. This can be useful for monitoring and ensuring that the backend is running properly.
@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "AI Assistant backend is running"
    }

# This endpoint allows users to upload files to the backend. The uploaded file is received as an UploadFile object, which provides access to the file's metadata and content. The endpoint returns a JSON response containing the filename and content type of the uploaded file.
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):

    file_content = await file.read()

    file_info = detect_file_type(
        file.filename,
        file.content_type
    )

    extracted_content = extract_content(
        file_content,
        file_info["extension"]
    )

    return {
        **file_info,
        "file_size": len(file_content),
        "extracted_content": extracted_content
    }