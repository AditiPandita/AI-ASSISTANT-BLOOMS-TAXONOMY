
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship

from app.database.connection import Base


# 1. Uploaded documents
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(100))
    file_size = Column(Integer)
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    pages = relationship(
        "Page",
        back_populates="document",
        cascade="all, delete-orphan"
    )

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )


# 2. Extracted document pages
class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )

    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    document = relationship(
        "Document",
        back_populates="pages"
    )


# 3. Document chunks and embeddings
class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )

    chunk_id = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    # Store the 384-dimensional embedding as JSON
    embedding = Column(JSON, nullable=False)

    document = relationship(
        "Document",
        back_populates="chunks"
    )


# 4. Chat history
class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True
    )

    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(JSON)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


# 5. Generated questions and Bloom levels
class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True
    )

    question_text = Column(Text, nullable=False)
    bloom_level = Column(String(30), nullable=False)
    marks = Column(Float, nullable=True)
    difficulty = Column(String(30), nullable=True)
    answer = Column(Text, nullable=True)

    source_chunk_ids = Column(JSON)
    source_pages = Column(JSON)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

# 6. Generated question papers and teacher approval
class QuestionPaper(Base):
    __tablename__ = "question_papers"

    id = Column(Integer, primary_key=True, index=True)

    title = Column(
        String(255),
        nullable=False,
        default="AI Generated Question Paper"
    )

    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True
    )

    # Preserve the exact order of selected questions
    question_ids = Column(JSON, nullable=False)

    total_questions = Column(Integer, default=0)
    total_marks = Column(Float, default=0)

    status = Column(
        String(30),
        nullable=False,
        default="Draft"
    )

    teacher_feedback = Column(Text, nullable=True)

    exam_details = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    approved_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
