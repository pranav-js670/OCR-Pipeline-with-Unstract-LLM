# import streamlit as st
# import requests
# import time

# st.set_page_config(page_title="Health Report OCR & Analysis")
# st.title("Health Report OCR & Analysis")

# uploaded_file = st.file_uploader("Upload PDF/image", type=["jpg", "jpeg", "png", "pdf"])
# if uploaded_file:
#     files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
#     with st.spinner("Processing document..."):
#         # 1. Upload
#         resp = requests.post("http://localhost:8000/api/upload", files=files)
#         job_id = resp.json()["job_id"]
#         # 2. Poll status
#         status = ""
#         while status not in ["processed", "failed"]:
#             time.sleep(2)
#             resp2 = requests.get(f"http://localhost:8000/api/status/{job_id}")
#             status = resp2.json()["status"]
#             st.write(f"Status: {status}")
#         # 3. Fetch results
#         if status == "processed":
#             resp3 = requests.get(f"http://localhost:8000/api/results/{job_id}")
#             data = resp3.json()["parameters"]
#             st.success("Analysis complete!")
#             st.json(data)
#         else:
#             st.error("Document processing failed.")

import streamlit as st
import requests
import time

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="Health Report OCR & Analysis")
st.title("Health Report OCR & Analysis")

uploaded_file = st.file_uploader("Upload PDF/Image", type=["pdf", "png", "jpg", "jpeg"])
if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    with st.spinner("Uploading and processing..."):
        # 1) POST /upload
        resp = requests.post(f"{API_BASE}/upload", files=files)
        if resp.status_code != 200:
            st.error(f"Upload failed: {resp.text}")
            st.stop()
        job_id = resp.json().get("job_id")
        if not job_id:
            st.error(f"Unexpected response from upload: {resp.json()}")
            st.stop()

        # 2) Poll /status/{job_id}
        status = ""
        while status not in ["processed", "failed"]:
            time.sleep(2)
            resp2 = requests.get(f"{API_BASE}/status/{job_id}")
            if resp2.status_code != 200:
                st.error(f"Status check failed: {resp2.text}")
                st.stop()
            status = resp2.json().get("status", "")
            st.write(f"Status: {status}")

        # 3) GET /results/{job_id}
        resp3 = requests.get(f"{API_BASE}/results/{job_id}")
        body = resp3.json()
        if resp3.status_code != 200:
            # Show the error JSON from the backend
            st.error(
                f"Error in results: {body.get('error', resp3.text)}\nDetail: {body.get('detail')}"
            )
            st.stop()

        params = body.get("parameters")
        if not isinstance(params, list):
            st.error(f"Unexpected results format: {body}")
            st.stop()

        # 4) Success—render JSON
        st.success("Analysis complete!")
        st.json(params)
