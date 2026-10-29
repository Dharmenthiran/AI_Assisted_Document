"""End-to-End Real Pipeline Verification Script.

Tests real PDF generation, file upload, PyMuPDF extraction, SentenceTransformers embedding,
ChromaDB persistent indexing, real Ollama llama3.2:3b generation, citation ground-truth,
and document deletion.
"""

import io
import sys
from pathlib import Path
import pymupdf  # Modern PyMuPDF API
from fastapi.testclient import TestClient

# Ensure app is in path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import app


def create_sample_pdf() -> bytes:
    """Generate a sample 2-page company policy PDF."""
    doc = pymupdf.open()

    # Page 1: Leave Policy
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "ACME CORP - LEAVE & VACATION POLICY (2026)\n\n"
        "1. Annual Leave: All full-time employees are granted 18 days of paid annual vacation.\n"
        "2. Sick Leave: Employees are entitled to 10 paid sick days per calendar year.\n"
        "3. Parental Leave: Maternity leave provides 16 weeks of 100% paid leave for primary caregivers.\n"
        "4. Carryover: Up to 5 unused annual leave days can roll over into the next year.",
        fontsize=12,
    )

    # Page 2: Remote Work & Equipment
    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "ACME CORP - REMOTE WORK & EQUIPMENT POLICY\n\n"
        "1. Hybrid Schedule: Eligible employees may work remotely up to 3 days per week.\n"
        "2. Home Office Stipend: The company provides a $1,200 annual equipment stipend for ergonomic desks, "
        "external monitors, and noise-canceling headsets.\n"
        "3. Security: All remote work must be conducted on company-managed devices with VPN enabled.",
        fontsize=12,
    )

    pdf_stream = io.BytesIO()
    doc.save(pdf_stream)
    doc.close()
    return pdf_stream.getvalue()


def run_verification():
    print("=================================================================")
    print("       STARTING REAL END-TO-END RAG VERIFICATION PIPELINE       ")
    print("=================================================================")

    with TestClient(app) as client:
        # Step 0: Ensure clean state
        existing = client.get("/api/documents").json().get("documents", [])
        for doc in existing:
            client.delete(f"/api/documents/{doc['document_id']}")

        # Step 1: Health check
        print("\n[Step 1] Checking /health endpoint...")
        health_res = client.get("/health")
        print(f"Health Response ({health_res.status_code}): {health_res.json()}")
        assert health_res.status_code == 200

        # Step 2: Upload Real PDF
        print("\n[Step 2] Uploading and Indexing real PDF...")
        pdf_bytes = create_sample_pdf()
        files = {"file": ("acme_policy_2026.pdf", pdf_bytes, "application/pdf")}
        upload_res = client.post("/api/documents/upload", files=files)
        print(f"Upload Response ({upload_res.status_code}): {upload_res.json()}")
        assert upload_res.status_code == 201
        upload_data = upload_res.json()
        doc_id = upload_data["document_id"]
        assert upload_data["page_count"] == 2
        assert upload_data["chunk_count"] >= 2
        print(f"-> Successfully indexed doc_id: {doc_id} with {upload_data['chunk_count']} chunks.")

        # Step 3: Query Document List
        print("\n[Step 3] Fetching document list...")
        list_res = client.get("/api/documents")
        print(f"Documents List ({list_res.status_code}): Total = {list_res.json()['total_count']}")
        assert list_res.status_code == 200
        assert list_res.json()["total_count"] >= 1

        # Step 4: Ask Question Grounded in Document (Remote Work Stipend)
        print("\n[Step 4] Asking Question 1 (Grounded in Page 2):")
        question1 = "What is the annual home office stipend and remote work schedule?"
        print(f"Question: '{question1}'")
        chat_res1 = client.post("/api/chat", json={"question": question1})
        print(f"Chat Response Code: {chat_res1.status_code}")
        data1 = chat_res1.json()
        print(f"Generated Answer:\n{data1.get('answer')}")
        print(f"\nRetrieved Citations:")
        for idx, src in enumerate(data1.get("sources", []), 1):
            print(f"  [{idx}] {src['filename']} — Page {src['page']} (Sim: {round(src['score']*100, 1)}%)")
            print(f"      Snippet: {src['snippet'][:120]}...")
        assert chat_res1.status_code == 200
        assert data1["context_used"] is True
        assert len(data1["sources"]) > 0
        # Verify citation page points to page 2 where the stipend is discussed
        assert any(s["page"] == 2 for s in data1["sources"])

        # Step 5: Ask Question with Missing Evidence
        print("\n[Step 5] Asking Question 2 (Evidence Not in Documents):")
        question2 = "What is the stock option vesting cliff schedule?"
        print(f"Question: '{question2}'")
        chat_res2 = client.post("/api/chat", json={"question": question2})
        print(f"Chat Response Code: {chat_res2.status_code}")
        data2 = chat_res2.json()
        print(f"Generated Answer:\n{data2.get('answer')}")
        assert chat_res2.status_code == 200

        # Step 6: Delete Document and purge chunks
        print("\n[Step 6] Deleting document and purging vector records...")
        del_res = client.delete(f"/api/documents/{doc_id}")
        print(f"Delete Response ({del_res.status_code}): {del_res.json()}")
        assert del_res.status_code == 200
        assert del_res.json()["deleted_chunks"] >= 2

        # Step 7: Verify clean state
        list_after = client.get("/api/documents").json()
        print(f"Documents count after deletion: {list_after['total_count']}")
        assert list_after["total_count"] == 0

        print("\n=================================================================")
        print("    REAL END-TO-END RAG PIPELINE VERIFICATION PASSED 100%!       ")
        print("=================================================================")


if __name__ == "__main__":
    run_verification()
