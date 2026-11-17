
import json

from app.database.connection import SessionLocal
from app.database.models import Page
from app.template_analyzer import analyze_exam_template


DOCUMENT_ID = 4

db = SessionLocal()

try:

    # STEP 1: Fetch uploaded document pages
    pages = (
        db.query(Page)
        .filter(Page.document_id == DOCUMENT_ID)
        .order_by(Page.page_number)
        .all()
    )

    if not pages:
        raise ValueError(
            f"No pages found for document ID {DOCUMENT_ID}"
        )

    page_data = [
        {
            "page": page.page_number,
            "text": page.text
        }
        for page in pages
    ]

    # STEP 2: Display extracted PDF text
    print("\n========== RAW EXTRACTED TEXT ==========\n")

    for page in page_data:
        print(f"\n--- PAGE {page['page']} ---\n")
        print(page["text"])

    print("\n========== END OF EXTRACTED TEXT ==========\n")

    # STEP 3: Analyze examination template
    print("\nSending document to Gemini...\n")

    blueprint = analyze_exam_template(
        pages=page_data,
        template_type="auto"
    )

    # STEP 4: Display blueprint
    print("\n========== EXAM BLUEPRINT ==========\n")

    print(
        json.dumps(
            blueprint,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\n========== ANALYSIS COMPLETED ==========")

except Exception as error:

    print("\nTEMPLATE ANALYSIS FAILED:")
    print(error)

finally:

    db.close()
