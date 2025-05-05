# import os
# import time
# import requests
# from dotenv import load_dotenv

# load_dotenv()

# BASE_URL = os.getenv("LLMWHISPERER_BASE_URL_V2")
# HEADERS = {"unstract-key": os.getenv("LLMWHISPERER_API_KEY")}


# # Submit document for processing
# def submit_document(file_bytes: bytes, filename: str) -> str:
#     response = requests.post(
#         f"{BASE_URL}/whisper",
#         headers=HEADERS,
#         files={"file": (filename, file_bytes)},
#         data={"mode": "form", "output_mode": "layout_peserving"},
#     )
#     response.raise_for_status()
#     return response.json()["whisper_hash"]


# # Check processing status
# def check_status(whsiper_hash: str) -> str:
#     response = requests.get(
#         f"{BASE_URL}/whisper-status",
#         headers=HEADERS,
#         params={"whisper_hash": whsiper_hash},
#     )
#     response.raise_for_status()
#     return response.json()["status"]


# # Retrive extracted text from the document after processing
# def retrieve_text(whisper_hash: str) -> str:
#     response = requests.get(
#         f"{BASE_URL}/whisper-retrieve",
#         headers=HEADERS,
#         params={"whisper_hash": whisper_hash, "text_only": "true"},
#     )
#     response.raise_for_status()
#     return response.json()["extraction"]

# backend/app/services/unstract.py

# import os
# import tempfile
# import time
# from unstract.llmwhisperer import LLMWhispererClientV2
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
# from dotenv import load_dotenv

# load_dotenv()

# # Initialize the official client (reads base URL & API key from env)
# client = LLMWhispererClientV2()


# def submit_document(file_bytes: bytes, filename: str) -> str:
#     # Write upload into a temp file so the client can read it
#     ext = os.path.splitext(filename)[1] or ".pdf"
#     with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
#         tmp.write(file_bytes)
#         tmp.flush()
#         tmp_path = tmp.name

#     try:
#         result = client.whisper(file_path=tmp_path)
#         return result["whisper_hash"]
#     finally:
#         try:
#             os.unlink(tmp_path)
#         except OSError:
#             pass


# def check_status(whisper_hash: str) -> str:
#     try:
#         status_resp = client.whisper_status(whisper_hash=whisper_hash)
#         return status_resp["status"]
#     except LLMWhispererClientException as e:
#         raise


# def retrieve_text(whisper_hash: str) -> str:
#     try:
#         retrieved = client.whisper_retrieve(whisper_hash=whisper_hash)
#         # client_v2 returns the extracted text under "extraction"
#         return retrieved.get("extraction") or retrieved.get("extracted_text", "")
#     except LLMWhispererClientException as e:
#         raise

# backend/app/services/unstract.py
# import os, tempfile
# from unstract.llmwhisperer import LLMWhispererClientV2
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
# from dotenv import load_dotenv

# load_dotenv()

# client = LLMWhispererClientV2()


# def submit_document(file_bytes: bytes, filename: str) -> str:
#     ext = os.path.splitext(filename)[1] or ".pdf"
#     with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
#         tmp.write(file_bytes)
#         tmp.flush()
#         tmp_path = tmp.name
#     try:
#         result = client.whisper(file_path=tmp_path)
#         return result["whisper_hash"]
#     finally:
#         os.unlink(tmp_path)


# def check_status(whisper_hash: str) -> str:
#     resp = client.whisper_status(whisper_hash=whisper_hash)
#     return resp["status"]


# def retrieve_text(whisper_hash: str) -> str:
#     resp = client.whisper_retrieve(whisper_hash=whisper_hash)
#     return resp.get("extraction") or resp.get("extracted_text", "")

# import os
# import tempfile
# from unstract.llmwhisperer import LLMWhispererClientV2
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
# from dotenv import load_dotenv
# from fastapi import HTTPException
# import logging

# load_dotenv()
# logger = logging.getLogger(__name__)

# client = LLMWhispererClientV2()


# def submit_document(file_bytes: bytes, filename: str) -> str:
#     ext = os.path.splitext(filename)[1] or ".pdf"
#     with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
#         tmp.write(file_bytes)
#         tmp.flush()
#         tmp_path = tmp.name
#     try:
#         result = client.whisper(file_path=tmp_path)
#         return result["whisper_hash"]
#     finally:
#         os.unlink(tmp_path)


# def check_status(whisper_hash: str) -> str:
#     resp = client.whisper_status(whisper_hash=whisper_hash)
#     return resp["status"]


# def retrieve_text(whisper_hash: str) -> str:
#     """
#     Retrieve OCR text from LLMWhisperer.
#     Uses the 'result_text' key in the JSON response – not 'extraction' or 'extracted_text' :contentReference[oaicite:0]{index=0}.
#     Raises HTTPException on failure or missing text.
#     """
#     try:
#         resp = client.whisper_retrieve(
#             whisper_hash=whisper_hash,
#             text_only=False,  # default; returns JSON with metadata and 'result_text'
#         )
#     except LLMWhispererClientException as e:
#         raise HTTPException(status_code=502, detail=str(e))

#     # Ensure the expected key is present
#     if "result_text" not in resp:
#         logger.error(f"Unexpected retrieve response: {resp!r}")
#         raise HTTPException(status_code=500, detail="No OCR text found in response.")

#     raw_text = resp["result_text"]
#     # Validate non-empty
#     if not raw_text.strip():
#         logger.error("Retrieved OCR text is empty.")
#         raise HTTPException(
#             status_code=500,
#             detail="OCR returned empty text—check your input document or API key.",
#         )

#     logger.debug(f"Raw OCR text (first 500 chars):\n{raw_text[:500]}…")
#     return raw_text

# import os
# import tempfile
# import logging
# from fastapi import HTTPException
# from dotenv import load_dotenv
# from unstract.llmwhisperer import LLMWhispererClientV2
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException

# load_dotenv()
# logger = logging.getLogger(__name__)

# # Initialize the V2 client (reads base_url and api_key from env by default)
# client = LLMWhispererClientV2()


# def submit_document(file_bytes: bytes, filename: str) -> str:
#     """
#     Submits a document for OCR. Returns the whisper_hash.
#     """
#     ext = os.path.splitext(filename)[1] or ".pdf"
#     with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
#         tmp.write(file_bytes)
#         tmp.flush()
#         tmp_path = tmp.name

#     try:
#         result = client.whisper(file_path=tmp_path, output_mode="layout_preserving")
#         # whisper() returns a dict with 'whisper_hash' per API :contentReference[oaicite:2]{index=2}
#         return result["whisper_hash"]
#     finally:
#         os.unlink(tmp_path)


# def check_status(whisper_hash: str) -> str:
#     """
#     Polls the OCR status endpoint. Returns status string.
#     """
#     resp = client.whisper_status(whisper_hash=whisper_hash)
#     return resp["status"]  # e.g., "processing", "processed"


# def retrieve_text(whisper_hash: str) -> str:
#     """
#     Retrieves the OCR’d text from LLMWhisperer V2.
#     Uses the 'extraction' key in the JSON response (not 'result_text') :contentReference[oaicite:3]{index=3}.
#     Raises HTTPException if retrieval fails or returns empty.
#     """
#     try:
#         # V2 whisper_retrieve signature: whisper_retrieve(whisper_hash: str) -> dict :contentReference[oaicite:4]{index=4}
#         resp = client.whisper_retrieve(whisper_hash=whisper_hash)
#     except LLMWhispererClientException as e:
#         logger.error(f"LLMWhisperer retrieval error: {e}")
#         raise HTTPException(status_code=502, detail=str(e))

#     # Extract text from the 'extraction' field
#     if "extraction" not in resp:
#         logger.error(f"Unexpected retrieve response (no 'extraction'): {resp!r}")
#         raise HTTPException(
#             status_code=500, detail="OCR response missing expected field."
#         )

#     raw_text = resp["extraction"]
#     if not isinstance(raw_text, str) or not raw_text.strip():
#         logger.error("OCR returned empty or invalid text.")
#         raise HTTPException(
#             status_code=500,
#             detail="OCR returned empty text—please verify the input document.",
#         )

#     logger.debug(f"Raw OCR text (first 500 chars):\n{raw_text[:500]}…")
#     return raw_text

# import os
# import tempfile
# import logging
# from fastapi import HTTPException
# from dotenv import load_dotenv
# from unstract.llmwhisperer import LLMWhispererClientV2
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException

# load_dotenv()
# logger = logging.getLogger(__name__)

# # Initialize the V2 client (reads base_url and api_key from env by default)
# client = LLMWhispererClientV2()


# def submit_document(file_bytes: bytes, filename: str) -> str:
#     """
#     Submits a document for OCR in layout-preserving mode and returns the whisper_hash.
#     """
#     ext = os.path.splitext(filename)[1] or ".pdf"
#     with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
#         tmp.write(file_bytes)
#         tmp.flush()
#         tmp_path = tmp.name

#     try:
#         # output_mode="layout_preserving" preserves the document’s original layout
#         result = client.whisper(file_path=tmp_path, output_mode="layout_preserving")
#         return result["whisper_hash"]
#     finally:
#         os.unlink(tmp_path)


# def check_status(whisper_hash: str) -> str:
#     """
#     Polls the OCR status endpoint. Returns a status string like "processing" or "processed".
#     """
#     resp = client.whisper_status(whisper_hash=whisper_hash)
#     return resp["status"]


# def retrieve_text(whisper_hash: str) -> str:
#     """
#     Retrieves the OCR’d text via whisper_retrieve().
#     Reads from the 'result_text' key (per V2 Retrieve API) and validates it. :contentReference[oaicite:1]{index=1}
#     Raises HTTPException if anything goes wrong or the text is empty.
#     """
#     try:
#         # Note: whisper_retrieve() in v2 has signature whisper_retrieve(whisper_hash)
#         resp = client.whisper_retrieve(whisper_hash=whisper_hash)
#     except LLMWhispererClientException as e:
#         logger.error(f"LLMWhisperer retrieval error: {e}")
#         raise HTTPException(status_code=502, detail=str(e))

#     # Extract the text
#     if "result_text" not in resp:
#         logger.error(f"Unexpected retrieve response (missing 'result_text'): {resp!r}")
#         raise HTTPException(
#             status_code=500, detail="OCR response missing expected field 'result_text'."
#         )

#     raw_text = resp["result_text"]
#     if not isinstance(raw_text, str) or not raw_text.strip():
#         logger.error("OCR returned empty or invalid text.")
#         raise HTTPException(
#             status_code=500,
#             detail="OCR returned empty text—please verify the input document or API key.",
#         )

#     logger.debug(f"Raw OCR text (first 500 chars):\n{raw_text[:500]}…")
#     return raw_text

# import os
# import tempfile
# import logging
# from fastapi import HTTPException
# from dotenv import load_dotenv
# from unstract.llmwhisperer import LLMWhispererClientV2
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException

# load_dotenv()
# logger = logging.getLogger(__name__)

# client = LLMWhispererClientV2()


# def submit_document(file_bytes: bytes, filename: str) -> str:
#     """
#     Submits a document for OCR in layout-preserving mode and returns the whisper_hash.
#     """
#     ext = os.path.splitext(filename)[1] or ".pdf"
#     with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
#         tmp.write(file_bytes)
#         tmp.flush()
#         tmp_path = tmp.name
#     try:
#         result = client.whisper(file_path=tmp_path, output_mode="layout_preserving")
#         return result["whisper_hash"]
#     finally:
#         os.unlink(tmp_path)


# def check_status(whisper_hash: str) -> str:
#     """
#     Polls the OCR status endpoint. Returns a status string like "processing" or "processed".
#     """
#     resp = client.whisper_status(whisper_hash=whisper_hash)
#     return resp["status"]


# def retrieve_text(whisper_hash: str) -> str:
#     """
#     Retrieves the OCR’d text via whisper_retrieve().
#     Tries multiple fields (result_text, extracted_text, or extraction if it's a string).
#     Raises HTTPException if none yields a non-empty string.
#     """
#     try:
#         resp = client.whisper_retrieve(whisper_hash=whisper_hash)
#     except LLMWhispererClientException as e:
#         logger.error(f"LLMWhisperer retrieval error: {e}")
#         raise HTTPException(status_code=502, detail=str(e))

#     # Try the main text fields in order
#     raw_text = None

#     # 1) Top-level 'result_text'
#     if isinstance(resp.get("result_text"), str):
#         raw_text = resp["result_text"]

#     # 2) Top-level 'extracted_text'
#     elif isinstance(resp.get("extracted_text"), str):
#         raw_text = resp["extracted_text"]

#     # 3) If 'extraction' is itself a string
#     elif isinstance(resp.get("extraction"), str):
#         raw_text = resp["extraction"]

#     # No valid text found?
#     if not raw_text or not raw_text.strip():
#         logger.error(
#             f"OCR response missing text fields or returned empty: {resp.keys()}"
#         )
#         raise HTTPException(
#             status_code=500,
#             detail="OCR returned empty text—please verify the input document or your API configuration.",
#         )

#     logger.debug(f"Raw OCR text (first 500 chars):\n{raw_text[:500]}…")
#     return raw_text

import os
import tempfile
import logging
from fastapi import HTTPException
from dotenv import load_dotenv
from unstract.llmwhisperer import LLMWhispererClientV2
from unstract.llmwhisperer.client_v2 import LLMWhispererClientException

load_dotenv()
logger = logging.getLogger(__name__)

client = LLMWhispererClientV2()


def submit_document(file_bytes: bytes, filename: str) -> str:
    """
    Submits a document for OCR in layout-preserving mode and returns the whisper_hash.
    """
    ext = os.path.splitext(filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

    try:
        result = client.whisper(file_path=tmp_path, output_mode="layout_preserving")
        return result["whisper_hash"]
    finally:
        os.unlink(tmp_path)


def check_status(whisper_hash: str) -> str:
    """
    Polls the OCR status endpoint. Returns a status string like "processing" or "processed".
    """
    resp = client.whisper_status(whisper_hash=whisper_hash)
    return resp["status"]


def retrieve_text(whisper_hash: str) -> str:
    """
    Retrieves the OCR’d text via whisper_retrieve().
    Extracts from extraction['result_text'] (per the v2 Retrieve API). :contentReference[oaicite:1]{index=1}
    Raises HTTPException if nothing valid is found.
    """
    try:
        resp = client.whisper_retrieve(whisper_hash=whisper_hash)
    except LLMWhispererClientException as e:
        logger.error(f"LLMWhisperer retrieval error: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    # The V2 client wraps the real payload under "extraction"
    extraction = resp.get("extraction")
    if not isinstance(extraction, dict):
        logger.error(
            f"Unexpected retrieve response (no extraction dict): {resp.keys()}"
        )
        raise HTTPException(
            status_code=500, detail="OCR response missing expected payload."
        )

    # Per docs: extraction['result_text'] holds the text when text_only=false :contentReference[oaicite:2]{index=2}
    raw_text = extraction.get("result_text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        logger.error("OCR returned empty or invalid text.")
        raise HTTPException(
            status_code=500,
            detail="OCR returned empty text—please verify the input document or your API configuration.",
        )

    logger.debug(f"Raw OCR text (first 500 chars):\n{raw_text[:500]}…")
    return raw_text
