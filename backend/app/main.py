
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query
)

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.ingestion.detector import detect_file_type
from app.ingestion.router import extract_content

from app.chunking import chunk_pages
from app.embeddings import generate_embeddings
from app.rag import answer_question, retrieve_context

from app.database.connection import (
    engine,
    SessionLocal,
    Base
)

from app.database import models

from app.database.models import (
    Document,
    Page,
    Chunk,
    ChatHistory,
    Question,
    QuestionPaper
)

from app.store import (
    vector_store,
    rebuild_vector_store
)

from app.generator import (
    generate_bloom_questions,
    BLOOM_LEVELS
)

from app.exporters import (
    generate_pdf,
    generate_docx
)


# ==========================================
# CONSTANTS
# ==========================================

DIFFICULTY_LEVELS = [
    "Easy",
    "Medium",
    "Hard",
    "Mixed"
]

SOURCE_MODES = [
    "existing",
    "new",
    "hybrid"
]


# ==========================================
# APPLICATION LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    try:

        total_vectors = rebuild_vector_store(db)

        print(
            f"FAISS restored successfully: "
            f"{total_vectors} vectors loaded."
        )

    finally:
        db.close()

    yield

    print("AI Teacher Assistant shutting down.")


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="AI Teacher Assistant",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==========================================
# REQUEST MODELS
# ==========================================

class ChatRequest(BaseModel):
    question: str


class BloomGenerationRequest(BaseModel):
    document_id: int
    bloom_levels: list[str]
    questions_per_level: int = 5
    marks_per_question: float = 2


class BloomLevelSetting(BaseModel):
    bloom_level: str
    question_count: int
    marks_per_question: float


class SmartPaperRequest(BaseModel):

    document_id: int

    bloom_levels: list[str]

    questions_per_level: int = 2

    marks_per_question: float = 5

    level_settings: list[BloomLevelSetting] | None = None

    difficulty: str = "Mixed"

    source_mode: str = "hybrid"


class SmartPaperExportRequest(BaseModel):

    # Export is tied to a saved QuestionPaper, never arbitrary question IDs.
    paper_id: int

    exam_details: dict | None = None

    include_answers: bool = False

    include_sources: bool = False


# ==========================================
# HELPER: VALIDATE BLOOM LEVELS
# ==========================================

def validate_bloom_levels(bloom_levels):

    if not bloom_levels:

        raise HTTPException(
            status_code=400,
            detail="Select at least one Bloom level."
        )

    if len(bloom_levels) != len(set(bloom_levels)):

        raise HTTPException(
            status_code=400,
            detail="Duplicate Bloom levels are not allowed."
        )

    invalid_levels = [
        level
        for level in bloom_levels
        if level not in BLOOM_LEVELS
    ]

    if invalid_levels:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid Bloom levels: {invalid_levels}"
        )


# ==========================================
# HELPER: BUILD BLOOM SETTINGS
# ==========================================

def build_level_settings(request: SmartPaperRequest):

    validate_bloom_levels(request.bloom_levels)

    if request.level_settings is None:

        if not 1 <= request.questions_per_level <= 20:

            raise HTTPException(
                status_code=400,
                detail="Questions per level must be between 1 and 20."
            )

        if request.marks_per_question <= 0:

            raise HTTPException(
                status_code=400,
                detail="Marks per question must be greater than zero."
            )

        return {
            level: {
                "question_count": request.questions_per_level,
                "marks_per_question": request.marks_per_question
            }
            for level in request.bloom_levels
        }

    settings = {}

    for item in request.level_settings:

        if item.bloom_level in settings:

            raise HTTPException(
                status_code=400,
                detail=f"Duplicate settings for {item.bloom_level}."
            )

        if item.bloom_level not in request.bloom_levels:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{item.bloom_level} is not in "
                    "the selected Bloom levels."
                )
            )

        if not 1 <= item.question_count <= 20:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Question count for {item.bloom_level} "
                    "must be between 1 and 20."
                )
            )

        if item.marks_per_question <= 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Marks for {item.bloom_level} "
                    "must be greater than zero."
                )
            )

        settings[item.bloom_level] = {
            "question_count": item.question_count,
            "marks_per_question": item.marks_per_question
        }

    missing_levels = [
        level
        for level in request.bloom_levels
        if level not in settings
    ]

    if missing_levels:

        raise HTTPException(
            status_code=400,
            detail=(
                "Individual settings are missing for: "
                f"{missing_levels}"
            )
        )

    return settings


# ==========================================
# HELPER: CHECK DIFFICULTY
# ==========================================

def matches_difficulty(question, requested_difficulty):

    if requested_difficulty == "Mixed":
        return True

    return question.difficulty == requested_difficulty


# ==========================================
# HELPER: CONVERT QUESTION TO DICTIONARY
# ==========================================

def question_to_dict(question):

    return {
        "id": question.id,
        "document_id": question.document_id,
        "question": question.question_text,
        "answer": question.answer,
        "bloom_level": question.bloom_level,
        "marks": question.marks,
        "difficulty": question.difficulty,
        "source_chunk_ids": question.source_chunk_ids or [],
        "source_pages": question.source_pages or [],
        "created_at": question.created_at
    }


# ==========================================
# HELPER: FETCH QUESTIONS IN SELECTED ORDER
# ==========================================

def fetch_questions_in_order(db, question_ids):

    records = (
        db.query(Question)
        .filter(Question.id.in_(question_ids))
        .all()
    )

    question_map = {
        item.id: item
        for item in records
    }

    missing_ids = [
        question_id
        for question_id in question_ids
        if question_id not in question_map
    ]

    if missing_ids:

        raise HTTPException(
            status_code=404,
            detail=f"Questions not found: {missing_ids}"
        )

    return [
        question_map[question_id]
        for question_id in question_ids
    ]


# ==========================================
# HEALTH CHECK API
# ==========================================

@app.get("/api/health")
def health_check():

    return {
        "status": "ok",
        "message": "AI Teacher Assistant backend is running"
    }


# ==========================================
# DOCUMENT UPLOAD API
# ==========================================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):

    db = SessionLocal()

    try:

        file_content = await file.read()

        if not file_content:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        file_info = detect_file_type(
            file.filename,
            file.content_type
        )

        extracted_content = extract_content(
            file_content,
            file_info["extension"]
        )

        pages = extracted_content.get("pages", [])

        if not pages:

            raise HTTPException(
                status_code=400,
                detail="No readable content found in the document."
            )

        chunks = chunk_pages(
            pages,
            chunk_size=800,
            chunk_overlap=100
        )

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail="Could not create chunks from the document."
            )

        embedded_chunks = generate_embeddings(chunks)

        document = Document(
            filename=file.filename,
            content_type=file.content_type,
            file_size=len(file_content),
            page_count=len(pages),
            chunk_count=len(chunks)
        )

        db.add(document)
        db.flush()

        page_records = [
            Page(
                document_id=document.id,
                page_number=page["page"],
                text=page["text"]
            )
            for page in pages
        ]

        db.add_all(page_records)

        faiss_chunks = []

        for chunk in embedded_chunks:

            chunk_record = Chunk(
                document_id=document.id,
                chunk_id=chunk["chunk_id"],
                page_number=chunk["page"],
                text=chunk["text"],
                embedding=chunk["embedding"].tolist()
            )

            db.add(chunk_record)
            db.flush()

            faiss_chunks.append({
                **chunk,
                "document_id": document.id,
                "database_chunk_id": chunk_record.id
            })

        db.commit()

        vector_store.add_chunks(faiss_chunks)

        return {
            "message": "Document processed and saved successfully.",
            "document_id": document.id,
            "filename": file.filename,
            "file_size": len(file_content),
            "pages": len(pages),
            "chunks": len(chunks),
            "vectors": vector_store.index.ntotal
        }

    except HTTPException:

        db.rollback()
        raise

    except Exception as error:

        db.rollback()

        print(f"Document upload error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Document processing failed."
        )

    finally:

        await file.close()
        db.close()


# ==========================================
# GET UPLOADED DOCUMENTS API
# ==========================================

@app.get("/api/documents")
def get_documents():

    db = SessionLocal()

    try:
        documents = (
            db.query(Document)
            .order_by(Document.id.desc())
            .all()
        )

        return {
            "total_documents": len(documents),
            "documents": [
                {
                    "id": document.id,
                    "filename": document.filename,
                    "page_count": document.page_count,
                    "chunk_count": document.chunk_count,
                    "file_size": document.file_size
                }
                for document in documents
            ]
        }

    finally:
        db.close()

# ==========================================
# CHAT API
# ==========================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if vector_store.index.ntotal == 0:

        raise HTTPException(
            status_code=400,
            detail="Please upload a document before asking questions."
        )

    db = SessionLocal()

    try:

        result = answer_question(
            request.question,
            vector_store,
            top_k=3
        )

        sources = [
            {
                "document_id": item.get("document_id"),
                "page": item["page"],
                "chunk_id": item["chunk_id"],
                "score": item["score"]
            }
            for item in result["results"]
        ]

        chat_record = ChatHistory(
            question=request.question,
            answer=result["answer"],
            sources=sources
        )

        db.add(chat_record)
        db.commit()

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": sources
        }

    except Exception as error:

        db.rollback()

        print(f"Chat error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the question."
        )

    finally:
        db.close()


# ==========================================
# BLOOM QUESTION GENERATION API
# ==========================================

@app.post("/api/bloom/generate")
def generate_bloom(request: BloomGenerationRequest):

    validate_bloom_levels(request.bloom_levels)

    if not 1 <= request.questions_per_level <= 20:

        raise HTTPException(
            status_code=400,
            detail="Questions per level must be between 1 and 20."
        )

    if request.marks_per_question <= 0:

        raise HTTPException(
            status_code=400,
            detail="Marks per question must be greater than zero."
        )

    if vector_store.index.ntotal == 0:

        raise HTTPException(
            status_code=400,
            detail="Please upload a document first."
        )

    db = SessionLocal()

    try:

        selected_document = (
            db.query(Document)
            .filter(Document.id == request.document_id)
            .first()
        )

        if not selected_document:

            raise HTTPException(
                status_code=404,
                detail="Selected document not found."
            )

        retrieval = retrieve_context(
            "Generate academic assessment questions from the uploaded educational material.",
            vector_store,
            top_k=10,
            document_id=request.document_id
        )

        context = retrieval["context"]

        if not context:

            raise HTTPException(
                status_code=400,
                detail="No relevant content found in the selected document."
            )

        questions = generate_bloom_questions(
            context=context,
            bloom_levels=request.bloom_levels,
            questions_per_level=request.questions_per_level,
            marks_per_question=request.marks_per_question
        )

        chunk_page_map = {
            item["database_chunk_id"]: item["page"]
            for item in retrieval["results"]
            if item.get("database_chunk_id") is not None
        }

        saved_questions = []

        for item in questions:

            source_ids = item["source_chunk_ids"]

            if not source_ids:

                raise RuntimeError(
                    "Generated question has no source chunks."
                )

            if not set(source_ids).issubset(chunk_page_map.keys()):

                raise RuntimeError(
                    "Question contains invalid source chunk IDs."
                )

            source_pages = sorted({
                chunk_page_map[chunk_id]
                for chunk_id in source_ids
            })

            if not source_pages:

                raise RuntimeError(
                    "No valid source pages found."
                )

            question = Question(
                document_id=selected_document.id,
                question_text=item["question"],
                bloom_level=item["bloom_level"],
                marks=request.marks_per_question,
                difficulty=item.get("difficulty"),
                answer=item["answer"],
                source_chunk_ids=source_ids,
                source_pages=source_pages
            )

            db.add(question)
            saved_questions.append(question)

        db.commit()

        for question in saved_questions:
            db.refresh(question)

        return {
            "message": "Bloom questions generated successfully.",
            "document_id": selected_document.id,
            "document_name": selected_document.filename,
            "total_questions": len(saved_questions),
            "selected_levels": request.bloom_levels,
            "questions": [
                {
                    "id": item.id,
                    "question": item.question_text,
                    "answer": item.answer,
                    "bloom_level": item.bloom_level,
                    "marks": item.marks,
                    "difficulty": item.difficulty,
                    "source_pages": item.source_pages,
                    "source_chunk_ids": item.source_chunk_ids
                }
                for item in saved_questions
            ]
        }

    except HTTPException:

        db.rollback()
        raise

    except Exception as error:

        db.rollback()

        print(f"Bloom generation error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Bloom question generation failed."
        )

    finally:
        db.close()


# ==========================================
# PHASE 2: SMART QUESTION PAPER GENERATION
# ==========================================

@app.post("/api/smart-paper/generate")
def generate_smart_paper(request: SmartPaperRequest):

    level_settings = build_level_settings(request)

    if request.difficulty not in DIFFICULTY_LEVELS:

        raise HTTPException(
            status_code=400,
            detail="Difficulty must be Easy, Medium, Hard or Mixed."
        )

    if request.source_mode not in SOURCE_MODES:

        raise HTTPException(
            status_code=400,
            detail="Source mode must be existing, new or hybrid."
        )

    if request.source_mode in ["new", "hybrid"]:

        if vector_store.index.ntotal == 0:

            raise HTTPException(
                status_code=400,
                detail="Please upload a document first."
            )

    db = SessionLocal()

    try:

        selected_document = (
            db.query(Document)
            .filter(Document.id == request.document_id)
            .first()
        )

        if not selected_document:

            raise HTTPException(
                status_code=404,
                detail="Selected document not found."
            )

        final_questions = []
        generated_questions = []
        existing_questions = []
        level_summary = []

        retrieval = None
        chunk_page_map = {}

        if request.source_mode in ["new", "hybrid"]:

            retrieval = retrieve_context(
                "Generate academic examination questions from the uploaded educational material.",
                vector_store,
                top_k=10,
                document_id=request.document_id
            )

            if not retrieval.get("context"):

                raise HTTPException(
                    status_code=400,
                    detail="No relevant content found in the selected document."
                )

            chunk_page_map = {
                item["database_chunk_id"]: item["page"]
                for item in retrieval["results"]
                if item.get("database_chunk_id") is not None
            }

        for bloom_level in request.bloom_levels:

            settings = level_settings[bloom_level]

            required_count = settings["question_count"]

            marks_per_question = settings["marks_per_question"]

            selected_for_level = []

            # ------------------------------
            # EXISTING QUESTION SELECTION
            # ------------------------------

            if request.source_mode in ["existing", "hybrid"]:

                available_questions = (
                    db.query(Question)
                    .filter(
                        Question.document_id == request.document_id,
                        Question.bloom_level == bloom_level
                    )
                    .order_by(Question.id.desc())
                    .all()
                )

                suitable_questions = [
                    question
                    for question in available_questions
                    if (
                        matches_difficulty(
                            question,
                            request.difficulty
                        )
                        and abs(
                            float(question.marks or 0)
                            - float(marks_per_question)
                        ) < 0.000001
                    )
                ]

                selected_existing = suitable_questions[:required_count]

                for question in selected_existing:

                    question_data = {
                        "id": question.id,
                        "question": question.question_text,
                        "answer": question.answer,
                        "bloom_level": question.bloom_level,
                        "marks": question.marks,
                        "difficulty": question.difficulty,
                        "source_pages": question.source_pages or [],
                        "source_chunk_ids": question.source_chunk_ids or [],
                        "origin": "existing"
                    }

                    selected_for_level.append(question_data)
                    existing_questions.append(question_data)

            # ------------------------------
            # MISSING QUESTION COUNT
            # ------------------------------

            missing_count = (
                required_count - len(selected_for_level)
            )

            if (
                request.source_mode == "existing"
                and missing_count > 0
            ):

                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Not enough suitable questions in the Question Bank.",
                        "bloom_level": bloom_level,
                        "required_questions": required_count,
                        "available_questions": len(selected_for_level),
                        "missing_questions": missing_count,
                        "difficulty": request.difficulty,
                        "marks_per_question": marks_per_question
                    }
                )

            # ------------------------------
            # GENERATE MISSING QUESTIONS
            # ------------------------------

            if (
                request.source_mode in ["new", "hybrid"]
                and missing_count > 0
            ):

                newly_generated = generate_bloom_questions(
                    context=retrieval["context"],
                    bloom_levels=[bloom_level],
                    questions_per_level=missing_count,
                    marks_per_question=marks_per_question
                )

                if len(newly_generated) != missing_count:

                    raise RuntimeError(
                        f"Expected {missing_count} questions for "
                        f"{bloom_level}, but received "
                        f"{len(newly_generated)}."
                    )

                for item in newly_generated:

                    source_ids = item.get("source_chunk_ids", [])

                    if not source_ids:

                        raise RuntimeError(
                            "Generated question has no source chunks."
                        )

                    if not set(source_ids).issubset(
                        chunk_page_map.keys()
                    ):

                        raise RuntimeError(
                            "Generated question contains invalid source IDs."
                        )

                    source_pages = sorted({
                        chunk_page_map[chunk_id]
                        for chunk_id in source_ids
                    })

                    if not source_pages:

                        raise RuntimeError(
                            "No valid source pages found."
                        )

                    if item.get("bloom_level") != bloom_level:

                        raise RuntimeError(
                            "Generated question has an unexpected Bloom level."
                        )

                    if not item.get("question") or not item.get("answer"):

                        raise RuntimeError(
                            "Generated question or answer is empty."
                        )

                    question_record = Question(
                        document_id=selected_document.id,
                        question_text=item["question"],
                        bloom_level=bloom_level,
                        marks=marks_per_question,
                        difficulty=item.get("difficulty"),
                        answer=item["answer"],
                        source_chunk_ids=source_ids,
                        source_pages=source_pages
                    )

                    db.add(question_record)

                    generated_questions.append(question_record)

                    selected_for_level.append({
                        "record": question_record,
                        "question": item["question"],
                        "answer": item["answer"],
                        "bloom_level": bloom_level,
                        "marks": marks_per_question,
                        "difficulty": item.get("difficulty"),
                        "source_pages": source_pages,
                        "source_chunk_ids": source_ids,
                        "origin": "generated"
                    })

            # ------------------------------
            # ADD TO FINAL PAPER
            # ------------------------------

            final_questions.extend(selected_for_level)

            level_summary.append({
                "bloom_level": bloom_level,
                "required_questions": required_count,
                "selected_questions": len(selected_for_level),
                "marks_per_question": marks_per_question,
                "total_marks": (
                    len(selected_for_level) * marks_per_question
                )
            })

        # ------------------------------
        # SAVE GENERATED QUESTIONS
        # ------------------------------

        db.commit()

        for question in generated_questions:
            db.refresh(question)

        for item in final_questions:

            if item.get("origin") == "generated":

                record = item["record"]

                item["id"] = record.id

                del item["record"]

            else:

                item.pop("record", None)

        # ------------------------------
        # CALCULATE TOTALS
        # ------------------------------

        total_questions = len(final_questions)

        total_marks = sum(
            float(item["marks"])
            for item in final_questions
        )
        # ------------------------------
        # SAVE QUESTION PAPER AS DRAFT
        # ------------------------------

        paper = QuestionPaper(
            title="AI Generated Question Paper",
            document_id=selected_document.id,
            question_ids=[
                item["id"] for item in final_questions
            ],
            total_questions=total_questions,
            total_marks=total_marks,
            status="Draft"
        )

        db.add(paper)
        db.commit()
        db.refresh(paper)
        return {
            "message": "Smart question paper generated successfully.",
             "paper_id": paper.id,
            "status": paper.status,
            "document_id": selected_document.id,
            "document_name": selected_document.filename,
            "source_mode": request.source_mode,
            "difficulty": request.difficulty,
            "selected_bloom_levels": request.bloom_levels,
            "total_questions": total_questions,
            "total_marks": total_marks,
            "existing_questions_used": len(existing_questions),
            "new_questions_generated": len(generated_questions),
            "level_summary": level_summary,
            "questions": final_questions
        }

    except HTTPException:

        db.rollback()
        raise

    except Exception as error:

        db.rollback()

        print(f"Smart paper generation error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Smart question paper generation failed."
        )

    finally:
        db.close()

# ==========================================
# TEACHER PAPER REVIEW APIs
# ==========================================

@app.get("/api/papers")
def get_all_papers():

    db = SessionLocal()

    try:
        papers = (
            db.query(QuestionPaper)
            .order_by(QuestionPaper.id.desc())
            .all()
        )

        return {
            "total_papers": len(papers),
            "papers": [
                {
                    "paper_id": paper.id,
                    "title": paper.title,
                    "document_id": paper.document_id,
                    "question_ids": paper.question_ids,
                    "total_questions": paper.total_questions,
                    "total_marks": paper.total_marks,
                    "status": paper.status,
                    "teacher_feedback": paper.teacher_feedback,
                    "created_at": paper.created_at
                }
                for paper in papers
            ]
        }

    finally:
        db.close()


@app.get("/api/papers/{paper_id}")
def get_paper_details(paper_id: int):

    db = SessionLocal()

    try:
        paper = (
            db.query(QuestionPaper)
            .filter(QuestionPaper.id == paper_id)
            .first()
        )

        if not paper:
            raise HTTPException(
                status_code=404,
                detail="Question paper not found."
            )

        records = fetch_questions_in_order(
            db,
            paper.question_ids
        )

        return {
            "paper_id": paper.id,
            "title": paper.title,
            "document_id": paper.document_id,
            "total_questions": paper.total_questions,
            "total_marks": paper.total_marks,
            "status": paper.status,
            "teacher_feedback": paper.teacher_feedback,
            "questions": [
                question_to_dict(question)
                for question in records
            ]
        }

    finally:
        db.close()

        
# ==========================================
# TEACHER APPROVAL WORKFLOW
# ==========================================

@app.post("/api/papers/{paper_id}/submit")
def submit_paper_for_approval(paper_id: int):

    db = SessionLocal()

    try:
        paper = db.query(QuestionPaper).filter(
            QuestionPaper.id == paper_id
        ).first()

        if not paper:
            raise HTTPException(
                status_code=404,
                detail="Question paper not found."
            )

        if paper.status not in ["Draft", "Changes Requested"]:
            raise HTTPException(
                status_code=400,
                detail="Only Draft or Changes Requested papers can be submitted."
            )

        paper.status = "Pending Approval"
        db.commit()

        return {
            "message": "Paper submitted for approval.",
            "paper_id": paper.id,
            "status": paper.status
        }

    finally:
        db.close()


@app.post("/api/papers/{paper_id}/approve")
def approve_paper(paper_id: int):

    from datetime import datetime, timezone

    db = SessionLocal()

    try:
        paper = db.query(QuestionPaper).filter(
            QuestionPaper.id == paper_id
        ).first()

        if not paper:
            raise HTTPException(
                status_code=404,
                detail="Question paper not found."
            )

        if paper.status != "Pending Approval":
            raise HTTPException(
                status_code=400,
                detail="Only pending papers can be approved."
            )

        paper.status = "Approved"
        paper.approved_at = datetime.now(timezone.utc)
        paper.teacher_feedback = None

        db.commit()

        return {
            "message": "Question paper approved successfully.",
            "paper_id": paper.id,
            "status": paper.status
        }

    finally:
        db.close()


@app.post("/api/papers/{paper_id}/request-changes")
def request_paper_changes(
    paper_id: int,
    feedback: str = Query(...)
):

    db = SessionLocal()

    try:
        paper = db.query(QuestionPaper).filter(
            QuestionPaper.id == paper_id
        ).first()

        if not paper:
            raise HTTPException(
                status_code=404,
                detail="Question paper not found."
            )

        if paper.status != "Pending Approval":
            raise HTTPException(
                status_code=400,
                detail="Only pending papers can receive change requests."
            )

        if not feedback.strip():
            raise HTTPException(
                status_code=400,
                detail="Feedback cannot be empty."
            )

        paper.status = "Changes Requested"
        paper.teacher_feedback = feedback

        db.commit()

        return {
            "message": "Changes requested successfully.",
            "paper_id": paper.id,
            "status": paper.status,
            "teacher_feedback": paper.teacher_feedback
        }

    finally:
        db.close()

# ==========================================
# QUESTION BANK API
# ==========================================

@app.get("/api/questions")
def get_questions(

    document_id: int | None = None,

    bloom_level: str | None = None,

    difficulty: str | None = None,

    limit: int = Query(
        default=20,
        ge=1,
        le=100
    ),

    offset: int = Query(
        default=0,
        ge=0
    )

):

    db = SessionLocal()

    try:

        query = db.query(Question)

        if document_id is not None:

            query = query.filter(
                Question.document_id == document_id
            )

        if bloom_level is not None:

            if bloom_level not in BLOOM_LEVELS:

                raise HTTPException(
                    status_code=400,
                    detail="Invalid Bloom level."
                )

            query = query.filter(
                Question.bloom_level == bloom_level
            )

        if difficulty is not None:

            if difficulty not in ["Easy", "Medium", "Hard"]:

                raise HTTPException(
                    status_code=400,
                    detail="Invalid difficulty level."
                )

            query = query.filter(
                Question.difficulty == difficulty
            )

        total_questions = query.count()

        questions = (
            query
            .order_by(Question.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return {
            "total_questions": total_questions,
            "limit": limit,
            "offset": offset,
            "questions": [
                question_to_dict(question)
                for question in questions
            ]
        }

    except HTTPException:
        raise

    except Exception as error:

        print(f"Question Bank error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve questions."
        )

    finally:
        db.close()


# ==========================================
# COMMON QUESTION EXPORT FUNCTION
# ==========================================

def export_questions(
    db,
    file_format: str,
    document_id: int | None,
    bloom_level: str | None,
    difficulty: str | None,
    question_ids: list[int] | None,
    include_answers: bool,
    include_sources: bool
):

    query = db.query(Question)

    if document_id is not None:

        query = query.filter(
            Question.document_id == document_id
        )

    if bloom_level is not None:

        if bloom_level not in BLOOM_LEVELS:

            raise HTTPException(
                status_code=400,
                detail="Invalid Bloom level."
            )

        query = query.filter(
            Question.bloom_level == bloom_level
        )

    if difficulty is not None:

        if difficulty not in ["Easy", "Medium", "Hard"]:

            raise HTTPException(
                status_code=400,
                detail="Invalid difficulty."
            )

        query = query.filter(
            Question.difficulty == difficulty
        )

    if question_ids is not None:

        if not question_ids:

            raise HTTPException(
                status_code=400,
                detail="Select at least one question."
            )

        if len(question_ids) != len(set(question_ids)):

            raise HTTPException(
                status_code=400,
                detail="Duplicate question IDs are not allowed."
            )

        query = query.filter(
            Question.id.in_(question_ids)
        )

    questions = (
        query
        .order_by(Question.id.asc())
        .all()
    )

    if (
        question_ids is not None
        and len(questions) != len(question_ids)
    ):

        raise HTTPException(
            status_code=404,
            detail="One or more selected questions were not found."
        )

    if not questions:

        raise HTTPException(
            status_code=404,
            detail="No questions found for the selected filters."
        )

    question_data = [
        {
            "id": item.id,
            "question": item.question_text,
            "answer": item.answer,
            "bloom_level": item.bloom_level,
            "marks": item.marks,
            "difficulty": item.difficulty,
            "source_pages": item.source_pages or [],
            "source_chunk_ids": item.source_chunk_ids or []
        }
        for item in questions
    ]

    title = "AI Generated Question Paper"

    if file_format == "pdf":

        output = generate_pdf(
            questions=question_data,
            include_answers=include_answers,
            include_sources=include_sources,
            title=title
        )

        media_type = "application/pdf"
        filename = "question_paper.pdf"

    else:

        output = generate_docx(
            questions=question_data,
            include_answers=include_answers,
            include_sources=include_sources,
            title=title
        )

        media_type = (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        )

        filename = "question_paper.docx"

    return StreamingResponse(
        output,
        media_type=media_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            )
        }
    )


# ==========================================
# QUESTION BANK PDF EXPORT
# ==========================================

@app.get("/api/questions/export/pdf")
def export_questions_pdf(

    document_id: int | None = None,

    bloom_level: str | None = None,

    difficulty: str | None = None,

    question_ids: list[int] | None = Query(default=None),

    include_answers: bool = False,

    include_sources: bool = False

):

    db = SessionLocal()

    try:

        return export_questions(
            db=db,
            file_format="pdf",
            document_id=document_id,
            bloom_level=bloom_level,
            difficulty=difficulty,
            question_ids=question_ids,
            include_answers=include_answers,
            include_sources=include_sources
        )

    finally:
        db.close()


# ==========================================
# QUESTION BANK DOCX EXPORT
# ==========================================

@app.get("/api/questions/export/docx")
def export_questions_docx(

    document_id: int | None = None,

    bloom_level: str | None = None,

    difficulty: str | None = None,

    question_ids: list[int] | None = Query(default=None),

    include_answers: bool = False,

    include_sources: bool = False

):

    db = SessionLocal()

    try:

        return export_questions(
            db=db,
            file_format="docx",
            document_id=document_id,
            bloom_level=bloom_level,
            difficulty=difficulty,
            question_ids=question_ids,
            include_answers=include_answers,
            include_sources=include_sources
        )

    finally:
        db.close()


# ==========================================
# APPROVED SMART PAPER EXPORT HELPERS
# ==========================================

def get_approved_paper_questions(db, paper_id: int):
    """Fetch a saved paper and its questions; export only after approval."""
    paper = (
        db.query(QuestionPaper)
        .filter(QuestionPaper.id == paper_id)
        .first()
    )

    if not paper:
        raise HTTPException(status_code=404, detail="Question paper not found.")

    if paper.status != "Approved":
        raise HTTPException(
            status_code=403,
            detail="This question paper cannot be exported until a teacher approves it."
        )

    if not paper.question_ids:
        raise HTTPException(
            status_code=400,
            detail="This question paper does not contain any questions."
        )

    records = fetch_questions_in_order(db, paper.question_ids)

    questions = [
        {
            "id": item.id,
            "question": item.question_text,
            "answer": item.answer,
            "bloom_level": item.bloom_level,
            "marks": item.marks,
            "difficulty": item.difficulty,
            "source_pages": item.source_pages or [],
            "source_chunk_ids": item.source_chunk_ids or []
        }
        for item in records
    ]
    return paper, questions


# ==========================================
# PHASE 3: APPROVED SMART PAPER PDF EXPORT
# ==========================================

@app.post("/api/smart-paper/export/pdf")
def export_smart_paper_pdf(request: SmartPaperExportRequest):
    db = SessionLocal()
    try:
        paper, questions = get_approved_paper_questions(db, request.paper_id)

        output = generate_pdf(
            questions=questions,
            include_answers=request.include_answers,
            include_sources=request.include_sources,
            title=paper.title or "AI Generated Question Paper",
            exam_details=request.exam_details,
            group_by_bloom=True
        )

        return StreamingResponse(
            output,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                f'attachment; filename="approved_question_paper_{paper.id}.pdf"'
            }
        )
    except HTTPException:
        raise
    except Exception as error:
        print(f"Approved smart PDF export error: {error}")
        raise HTTPException(
            status_code=500,
            detail="Approved question paper PDF export failed."
        )
    finally:
        db.close()


# ==========================================
# PHASE 3: APPROVED SMART PAPER DOCX EXPORT
# ==========================================

@app.post("/api/smart-paper/export/docx")
def export_smart_paper_docx(request: SmartPaperExportRequest):
    db = SessionLocal()
    try:
        paper, questions = get_approved_paper_questions(db, request.paper_id)

        output = generate_docx(
            questions=questions,
            include_answers=request.include_answers,
            include_sources=request.include_sources,
            title=paper.title or "AI Generated Question Paper",
            exam_details=request.exam_details,
            group_by_bloom=True
        )

        return StreamingResponse(
            output,
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            headers={
                "Content-Disposition":
                f'attachment; filename="approved_question_paper_{paper.id}.docx"'
            }
        )
    except HTTPException:
        raise
    except Exception as error:
        print(f"Approved smart DOCX export error: {error}")
        raise HTTPException(
            status_code=500,
            detail="Approved question paper DOCX export failed."
        )
    finally:
        db.close()
