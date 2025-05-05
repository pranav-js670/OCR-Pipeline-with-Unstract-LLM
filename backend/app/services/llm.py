# import json
# import os
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# prompt = ChatPromptTemplate(
#     [
#         (
#             "system",
#             "You are a healthcare assistant. Extract lab test results from the given report.",
#         ),
#         (
#             "human",
#             (
#                 "Convert the following medical report into a JSON array. "
#                 "Each item must include: parameter (string), value (number), unit (string), "
#                 "reference_range (array of two numbers), assessment ('low', 'normal', or 'high').\n\n"
#                 "Report:\n{text}"
#             ),
#         ),
#     ]
# )

# llm = ChatGroq(
#     model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY")
# )

# chain = prompt | llm


# def parse_health_report(text: str) -> list:
#     response = chain.invoke({"text": text})
#     return json.loads(response.content)

# backend/app/services/llm.py

# import os
# import json
# from pydantic import BaseModel, Field
# from langchain_core.output_parsers.pydantic import (
#     PydanticOutputParser,
# )  # :contentReference[oaicite:0]{index=0}
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv

# load_dotenv()  # ensures GROQ_API_KEY is in env


# # 1) Pydantic model for each lab parameter
# class LabParameter(BaseModel):
#     parameter: str = Field(..., description="The name of the lab test")
#     value: float = Field(..., description="The numeric result value")
#     unit: str = Field(..., description="Unit of measurement")
#     reference_range: list[float] = Field(
#         ..., description="Two-element list: [low, high] bounds"
#     )
#     assessment: str = Field(..., description="One of 'low', 'normal', or 'high'")


# # 2) Create the parser from that model
# parser = PydanticOutputParser(
#     pydantic_object=LabParameter
# )  # :contentReference[oaicite:1]{index=1}

# # 3) Build the prompt template, injecting `parser.get_format_instructions()`
# prompt = ChatPromptTemplate(
#     [
#         (
#             "system",
#             "You are a healthcare assistant. Extract lab test results from the given report.",
#         ),
#         (
#             "user",
#             (
#                 "Convert the following medical report into a JSON array, "
#                 "where each entry follows this schema:\n{format_instructions}\n\n"
#                 "Report:\n{text}"
#             ),
#         ),
#     ]
# )

# # Bind the format instructions into the final prompt
# prompt = prompt.partial(
#     format_instructions=parser.get_format_instructions()
# )  # :contentReference[oaicite:2]{index=2}

# # 4) Instantiate the Groq client **with your API key**
# llm = ChatGroq(
#     model="llama-3.1-8b-instant",
#     api_key=os.getenv("GROQ_API_KEY"),  # ← critical fix: supply your key
#     temperature=0,
# )


# def parse_health_report(text: str) -> list[dict]:
#     """Invoke Groq + parser to return validated list of LabParameter dicts."""
#     # 5) Run the chat + parse step in one go
#     #    The parser ensures the LLM output is valid JSON matching LabParameter.
#     result = prompt | llm | parser  # :contentReference[oaicite:3]{index=3}

#     # 6) `result` is already a LabParameter instance or raises an error.
#     #    If you want raw dicts:
#     return [item.dict() for item in result]

# backend/app/services/llm.py

# from pydantic import BaseModel, Field
# from typing import List


# class LabParameter(BaseModel):
#     parameter: str = Field(..., description="Name of the lab test")
#     value: float = Field(..., description="Numeric result")
#     unit: str = Field(..., description="Unit of measurement")
#     reference_range: List[float] = Field(
#         ..., description="Two-element list [low, high]"
#     )
#     assessment: str = Field(..., description="One of 'low', 'normal', or 'high'")


# class LabReport(BaseModel):
#     """Wrapper model that holds a list of LabParameter items."""

#     parameters: List[LabParameter]


# from langchain_core.output_parsers.pydantic import PydanticOutputParser
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# import os

# # 1) Prepare the parser & prompt
# parser = PydanticOutputParser(pydantic_object=LabReport)
# prompt = ChatPromptTemplate(
#     [
#         ("system", "You are a healthcare assistant. Extract lab test results."),
#         (
#             "user",
#             (
#                 "Convert the following medical report into JSON matching this schema:\n"
#                 "{format_instructions}\n\nReport:\n{text}"
#             ),
#         ),
#     ]
# ).partial(format_instructions=parser.get_format_instructions())

# # 2) Instantiate your LLM with the API key
# llm = ChatGroq(
#     model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0
# )


# def parse_health_report(text: str) -> list[dict]:
#     # 3) Invoke the LLM to get raw text
#     llm_input = prompt.format_prompt(text=text)
#     llm_response = llm.invoke(llm_input.to_messages())

#     # 4) Parse the raw string into LabReport
#     report: LabReport = parser.parse(llm_response.content)
#     # 5) Return plain dicts
#     return [item.model_dump() for item in report.parameters]

# backend/app/services/llm.py

# import os
# from typing import List, Dict
# from unstract.llmwhisperer import LLMWhispererClientV2
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain_core.output_parsers.structured import (
#     StructuredOutputParser,
#     ResponseSchema,
# )
# from fastapi import HTTPException

# # Initialize Groq client
# groq = ChatGroq(
#     model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0
# )

# # 1️⃣ Define the response schema: one array field named "parameters"
# response_schemas = [
#     ResponseSchema(
#         name="parameters",
#         description=(
#             "A list of lab test results, where each item has: "
#             "parameter (string), value (number), unit (string), "
#             "reference_range (two-element list of numbers), assessment (one of 'low','normal','high')"
#         ),
#     )
# ]

# # 2️⃣ Create the StructuredOutputParser
# parser = StructuredOutputParser.from_response_schemas(
#     response_schemas
# )  # :contentReference[oaicite:2]{index=2}

# # 3️⃣ Build the prompt, injecting the parser instructions
# prompt = ChatPromptTemplate(
#     [
#         (
#             "system",
#             "You are a healthcare assistant. Extract lab test results from the report.",
#         ),
#         (
#             "user",
#             (
#                 "Respond **only** with a JSON object matching this schema:\n\n"
#                 "{format_instructions}\n\n"
#                 "Medical report:\n{text}"
#             ),
#         ),
#     ]
# ).partial(format_instructions=parser.get_format_instructions())


# def parse_health_report(text: str) -> List[Dict]:
#     """
#     Extracts and parses lab parameters from raw text via Groq + StructuredOutputParser.
#     """
#     try:
#         # 4️⃣ Format the prompt and invoke the LLM
#         msg = prompt.format_prompt(text=text).to_messages()
#         llm_resp = groq.invoke(msg)

#         # 5️⃣ Parse and return the JSON
#         parsed = parser.parse(llm_resp.content)
#         # parsed is a dict: {"parameters": [...]}
#         return parsed["parameters"]

#     except HTTPException:
#         raise
#     except LLMWhispererClientException as e:
#         raise HTTPException(status_code=502, detail=f"LLM error: {e}")
#     except Exception as e:
#         # Bubble up parse errors cleanly
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to parse JSON output:\n{llm_resp.content}\nError: {e}",
#         )

# backend/app/services/llm.py

# from pydantic import BaseModel, Field
# from typing import List


# class LabParameter(BaseModel):
#     parameter: str = Field(..., description="Name of the lab test")
#     value: float = Field(..., description="Numeric result")
#     unit: str = Field(..., description="Unit of measurement")
#     reference_range: List[float] = Field(
#         ..., description="Two-element list [low, high]"
#     )
#     assessment: str = Field(..., description="One of 'low','normal','high'")


# class LabReport(BaseModel):
#     """Wrapper model for the entire report."""

#     parameters: List[LabParameter]


# from pydantic_ai import Agent, RunContext

# # Create an agent that speaks to your chosen LLM (Groq in this example)
# agent = Agent(
#     "groq:llama-3.1-8b-instant",  # PydanticAI’s Groq URI
#     result_type=LabReport,  # Enforce LabReport schema
#     system_prompt=(
#         "You are a healthcare assistant. "
#         "Extract lab test results from the given medical report."
#     ),
#     temperature=0.0,
# )

# import os
# from fastapi import HTTPException

# # Your existing Unstract import remains:
# from unstract.llmwhisperer import LLMWhispererClientV2
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException


# # Function to call the agent
# def parse_health_report(text: str) -> List[dict]:
#     """
#     Uses PydanticAI to extract and validate lab parameters from raw text.
#     Returns a list of dicts matching LabParameter.schema().
#     """
#     try:
#         # Run the agent synchronously
#         report: LabReport = agent.run(
#             input=text, context=RunContext(vars={"text": text})
#         )
#         # Return plain dicts
#         return [param.model_dump() for param in report.parameters]

#     except LLMWhispererClientException as e:
#         # If the Unstract client fails (unlikely here), propagate
#         raise HTTPException(status_code=502, detail=str(e))
#     except Exception as e:
#         # Anything else (LLM errors, schema mismatches)
#         raise HTTPException(status_code=500, detail=f"PydanticAI parsing error: {e}")

# backend/app/services/llm.py

# import os
# from typing import List, Dict

# from fastapi import HTTPException
# from pydantic import BaseModel, Field

# from pydantic_ai import Agent  # :contentReference[oaicite:0]{index=0}
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException


# # ─── Define your schemas ─────────────────────────────────────────────────────────
# class LabParameter(BaseModel):
#     parameter: str = Field(..., description="Name of the lab test")
#     value: float = Field(..., description="Numeric result")
#     unit: str = Field(..., description="Unit of measurement")
#     reference_range: List[float] = Field(
#         ..., description="Two-element list [low, high]"
#     )
#     assessment: str = Field(..., description="One of 'low','normal','high'")


# class LabReport(BaseModel):
#     parameters: List[LabParameter]


# # ─── Instantiate a PydanticAI Agent ─────────────────────────────────────────────
# agent = Agent(
#     "groq:llama-3.1-8b-instant",  # Use your Groq model URI :contentReference[oaicite:1]{index=1}
#     system_prompt=(
#         "You are a healthcare assistant. Extract lab test results from the given report."
#     ),
#     result_type=LabReport,  # Enforce the LabReport schema :contentReference[oaicite:2]{index=2}
#     temperature=0.0,
# )


# # ─── The parsing function ───────────────────────────────────────────────────────
# def parse_health_report(raw_text: str) -> List[Dict]:
#     """
#     Uses PydanticAI to extract validated LabReport from raw text.
#     Returns a list of dicts for each LabParameter.
#     """
#     try:
#         # Run the agent synchronously with the raw text
#         run_result = agent.run_sync(raw_text)  # :contentReference[oaicite:3]{index=3}
#         report: LabReport = run_result.data  # .data is your parsed Pydantic model

#         return [param.dict() for param in report.parameters]

#     except LLMWhispererClientException as e:
#         # If you tie in Unstract calls, map its errors here
#         raise HTTPException(status_code=502, detail=str(e))

#     except Exception as e:
#         # This catches any schema mismatch or LLM error
#         raise HTTPException(status_code=500, detail=f"PydanticAI parsing error: {e}")

# import os
# from typing import List, Dict
# from fastapi import HTTPException
# from pydantic import BaseModel, Field

# from pydantic_ai import Agent
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException


# # ─── SCHEMAS ────────────────────────────────────────────────────────────────────
# class LabParameter(BaseModel):
#     parameter: str = Field(..., description="Name of the lab test")
#     value: float = Field(..., description="Numeric result")
#     unit: str = Field(..., description="Unit of measurement")
#     reference_range: List[float] = Field(
#         ..., description="Two-element list [low, high]"
#     )
#     assessment: str = Field(..., description="One of 'low','normal','high'")


# class LabReport(BaseModel):
#     parameters: List[LabParameter]


# # ─── PydanticAI AGENT ──────────────────────────────────────────────────────────
# agent = Agent(
#     # Groq model URI; replace if needed
#     model="groq:llama-3.1-8b-instant",
#     system_prompt=(
#         "You are a strict healthcare assistant. You will ONLY extract lab test "
#         "parameters that are explicitly present in the medical report. "
#         "Do NOT invent any parameters."
#     ),
#     result_type=LabReport,
#     temperature=0.0,
# )


# # ─── PARSING FUNCTION ──────────────────────────────────────────────────────────
# def parse_health_report(raw_text: str) -> List[Dict]:
#     """
#     Extracts validated LabReport from raw_text via PydanticAI.
#     Returns a list of dicts for each LabParameter found.
#     """
#     try:
#         # Synchronously run the agent, passing raw_text as the only input
#         run_result = agent.run_sync(raw_text)
#         report: LabReport = run_result.data

#         # If nothing found, PydanticAI will give parameters=[]
#         return [p.dict() for p in report.parameters]

#     except LLMWhispererClientException as e:
#         # Unstract client errors propagate as 502
#         raise HTTPException(status_code=502, detail=str(e))

#     except Exception as e:
#         # Parsing or schema mismatch errors
#         raise HTTPException(
#             status_code=500,
#             detail=f"PydanticAI parsing error: {e}\nRaw text excerpt: {raw_text[:200]}...",
#         )


# backend/app/services/llm.py

# import re
# from typing import List, Dict
# from fastapi import HTTPException

# # No more PydanticAI or LangChain dependencies here!


# def parse_health_report(raw_text: str) -> List[Dict]:
#     """
#     Deterministically extract lab parameters from the raw OCR text.
#     Looks for the table headed by 'TEST PARAMETER' and 'RESULT'.
#     """
#     lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

#     # 1) Locate the header row
#     header_idx = None
#     for i, line in enumerate(lines):
#         if re.search(r"TEST\s+PARAMETER.*RESULT", line, flags=re.IGNORECASE):
#             header_idx = i
#             break

#     if header_idx is None:
#         # No table found
#         return []

#     results = []
#     # 2) Iterate each subsequent line until blank or end
#     for line in lines[header_idx + 1 :]:
#         # Stop if we hit a non-data line (e.g. 'NOTE:' or end)
#         if re.match(r"NOTE[:\s]", line, flags=re.IGNORECASE):
#             break

#         # Expect: <Parameter>   <Value>   <Unit>   <Optional Reference Range>
#         # We capture parameter (non‐digits), value (number), unit (word), and range (e.g. 10 - 16)
#         m = re.match(
#             r"(?P<parameter>[\w\s\-\/]+?)\s+"
#             r"(?P<value>\d+(\.\d+)?)\s+"
#             r"(?P<unit>[^\d\s]+)"
#             r"(?:\s+(?P<low>\d+(\.\d+)?)\s*-\s*(?P<high>\d+(\.\d+)?))?",
#             line,
#         )
#         if not m:
#             continue  # skip non-matching lines

#         param = m.group("parameter").strip()
#         value = float(m.group("value"))
#         unit = m.group("unit").strip()

#         # If no explicit reference range, set to [None,None]
#         if m.group("low") and m.group("high"):
#             low = float(m.group("low"))
#             high = float(m.group("high"))
#         else:
#             low = None
#             high = None

#         # 3) Determine assessment
#         if low is not None and high is not None:
#             if value < low:
#                 assessment = "low"
#             elif value > high:
#                 assessment = "high"
#             else:
#                 assessment = "normal"
#         else:
#             assessment = "normal"

#         results.append(
#             {
#                 "parameter": param,
#                 "value": value,
#                 "unit": unit,
#                 "reference_range": [low, high],
#                 "assessment": assessment,
#             }
#         )

#     return results

# import os
# import logging
# from typing import List, Dict
# from fastapi import HTTPException
# from pydantic import BaseModel, Field
# from pydantic_ai import Agent
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
# from dotenv import load_dotenv

# load_dotenv()


# # ─── SCHEMAS ────────────────────────────────────────────────────────────────────
# class LabParameter(BaseModel):
#     parameter: str = Field(..., description="Name of the lab test")
#     value: float = Field(..., description="Numeric result")
#     unit: str = Field(..., description="Unit of measurement")
#     reference_range: List[float] = Field(
#         ..., description="Two-element list [low, high]"
#     )
#     assessment: str = Field(..., description="One of 'low','normal','high'")


# class LabReport(BaseModel):
#     parameters: List[LabParameter]


# # ─── AGENT SETUP ────────────────────────────────────────────────────────────────
# agent = Agent(
#     model="groq:llama-3.1-8b-instant",
#     system_prompt=(
#         "You are a strict healthcare assistant. You will ONLY extract lab test "
#         "parameters that are explicitly present in the medical report. "
#         "Do NOT invent any parameters."
#     ),
#     result_type=LabReport,
#     temperature=0.0,
# )

# logger = logging.getLogger(__name__)


# # ─── PARSING FUNCTION ────────────────────────────────────────────────────────────
# def parse_health_report(raw_text: str) -> List[Dict]:
#     """
#     Passes validated raw_text to PydanticAI and returns a list of dicts.
#     Raises HTTPException on schema errors or if the LLM fails.
#     """
#     try:
#         logger.debug(f"Sending to PydanticAI agent. Text length: {len(raw_text)}")
#         run_result = agent.run_sync(raw_text)
#         report: LabReport = run_result.data
#         logger.debug(f"PydanticAI returned: {report.json()[:500]}…")
#         return [p.dict() for p in report.parameters]

#     except LLMWhispererClientException as e:
#         # Should only occur if PydanticAI tries calling Unstract again
#         raise HTTPException(status_code=502, detail=str(e))

#     except Exception as e:
#         # Catch schema validation or other errors
#         logger.exception("PydanticAI parsing error")
#         raise HTTPException(
#             status_code=500,
#             detail=f"PydanticAI parsing error: {e}\nRaw text excerpt: {raw_text[:200]}...",
#         )

# import logging
# from typing import List, Dict
# from fastapi import HTTPException
# from pydantic import BaseModel, Field
# from pydantic_ai import Agent
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
# from dotenv import load_dotenv

# load_dotenv()


# # ─── SCHEMAS ────────────────────────────────────────────────────────────────────
# class LabParameter(BaseModel):
#     parameter: str = Field(..., description="Name of the lab test")
#     value: float = Field(..., description="Numeric result")
#     unit: str = Field(..., description="Unit of measurement")
#     reference_range: List[float] = Field(
#         ..., description="Two-element list [low, high]"
#     )
#     assessment: str = Field(..., description="One of 'low','normal','high'")


# class LabReport(BaseModel):
#     parameters: List[LabParameter]


# # ─── AGENT SETUP ────────────────────────────────────────────────────────────────
# agent = Agent(
#     model="groq:llama-3.1-8b-instant",
#     system_prompt=(
#         "You are a strict healthcare assistant. You will ONLY extract lab test "
#         "parameters that are explicitly present in the medical report. "
#         "Do NOT invent any parameters."
#     ),
#     result_type=LabReport,
#     temperature=0.0,
# )

# logger = logging.getLogger(__name__)


# # ─── PARSING FUNCTION ────────────────────────────────────────────────────────────
# def parse_health_report(raw_text: str) -> List[Dict]:
#     """
#     Sends the OCR’d text to PydanticAI and returns a list of dicts.
#     Raises HTTPException on errors or schema mismatches.
#     """
#     try:
#         logger.debug(f"Sending to PydanticAI agent (text length: {len(raw_text)}).")
#         run_result = agent.run_sync(raw_text)
#         report: LabReport = run_result.data
#         logger.debug(f"PydanticAI returned {len(report.parameters)} parameters.")
#         return [param.dict() for param in report.parameters]

#     except LLMWhispererClientException as e:
#         logger.error(f"PydanticAI agent client error: {e}")
#         raise HTTPException(status_code=502, detail=str(e))

#     except Exception as e:
#         logger.exception("PydanticAI parsing error")
#         raise HTTPException(
#             status_code=500,
#             detail=f"PydanticAI parsing error: {e}\nExcerpt: {raw_text[:200]}…",
#         )

# import logging
# from typing import List, Dict, Optional
# from fastapi import HTTPException
# from pydantic import BaseModel, Field
# from pydantic_ai import Agent
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
# from dotenv import load_dotenv

# load_dotenv()


# # ─── SCHEMAS ────────────────────────────────────────────────────────────────────
# class LabParameter(BaseModel):
#     parameter: str = Field(..., description="Name of the lab test")
#     value: float = Field(..., description="Numeric result")
#     unit: str = Field(..., description="Unit of measurement")
#     reference_range: List[Optional[float]] = Field(
#         ..., description="Two-element list [low, high], with null for missing ends"
#     )
#     assessment: str = Field(..., description="One of 'low','normal','high'")


# class LabReport(BaseModel):
#     parameters: List[LabParameter]


# # ─── AGENT SETUP ────────────────────────────────────────────────────────────────
# agent = Agent(
#     model="groq:llama-3.1-8b-instant",
#     system_prompt=(
#         "You are a strict healthcare assistant. You will ONLY extract lab test "
#         "parameters that are explicitly present in the medical report. "
#         "Output a JSON array of objects with keys [parameter, value, unit, reference_range, assessment]. "
#         "Use JSON `null` (not the word ‘none’) for any missing numeric values. "
#         "Do NOT invent any parameters or ranges."
#     ),
#     result_type=LabReport,
#     temperature=0.0,
# )

# logger = logging.getLogger(__name__)


# # ─── PARSING FUNCTION ────────────────────────────────────────────────────────────
# def parse_health_report(raw_text: str) -> List[Dict]:
#     """
#     Sends the OCR’d text to PydanticAI and returns a list of dicts.
#     Raises HTTPException on errors or schema mismatches.
#     """
#     try:
#         logger.debug(f"Sending to PydanticAI agent (text length: {len(raw_text)})")
#         run_result = agent.run_sync(raw_text)
#         report: LabReport = run_result.data
#         logger.debug(f"PydanticAI returned {len(report.parameters)} parameters.")
#         return [param.dict() for param in report.parameters]

#     except LLMWhispererClientException as e:
#         logger.error(f"PydanticAI agent client error: {e}")
#         raise HTTPException(status_code=502, detail=str(e))

#     except Exception as e:
#         logger.exception("PydanticAI parsing error")
#         raise HTTPException(
#             status_code=500,
#             detail=f"PydanticAI parsing error: {e}\nExcerpt: {raw_text[:200]}…",
#         )

# import logging
# from typing import List, Dict, Optional
# from fastapi import HTTPException
# from pydantic import BaseModel, Field
# from pydantic_ai import Agent
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
# from pydantic_ai.exceptions import ModelHTTPError
# from dotenv import load_dotenv

# load_dotenv()
# import json
# import re
# import ast


# # ─── SCHEMAS ────────────────────────────────────────────────────────────────────
# class LabParameter(BaseModel):
#     parameter: str = Field(..., description="Name of the lab test")
#     value: float = Field(..., description="Numeric result")
#     unit: str = Field(..., description="Unit of measurement")
#     reference_range: List[Optional[float]] = Field(
#         ..., description="Two-element list [low, high], with null for missing ends"
#     )
#     assessment: str = Field(..., description="One of 'low','normal','high'")


# class LabReport(BaseModel):
#     parameters: List[LabParameter]


# # ─── AGENT SETUP ────────────────────────────────────────────────────────────────
# agent = Agent(
#     model="groq:llama-3.1-8b-instant",
#     system_prompt=(
#         "You are a strict healthcare assistant. You will ONLY extract lab test "
#         "parameters that are explicitly present in the medical report. "
#         "Output a JSON array of objects with keys [parameter, value, unit, reference_range, assessment]. "
#         "Use JSON null for missing numeric values. Do NOT invent any parameters or ranges."
#     ),
#     result_type=LabReport,
#     temperature=0.0,
# )

# logger = logging.getLogger(__name__)


# # ─── PARSING FUNCTION ────────────────────────────────────────────────────────────
# def parse_health_report(raw_text: str) -> List[Dict]:
#     """
#     Sends the OCR’d text to PydanticAI and returns a list of dicts.
#     Handles ModelHTTPError by extracting JSON from failed_generation and post-processing.
#     Raises HTTPException on unrecoverable errors.
#     """
#     try:
#         logger.debug(f"Sending to PydanticAI agent (text length: {len(raw_text)})")
#         run_result = agent.run_sync(raw_text)
#         report: LabReport = run_result.data
#         logger.debug(f"PydanticAI returned {len(report.parameters)} parameters.")
#         return [param.dict() for param in report.parameters]

#     except ModelHTTPError as e:
#         body = getattr(e, "body", {})
#         failed = body.get("error", {}).get("failed_generation", "")
#         # Attempt to extract JSON substring after final_result>
#         match = re.search(r"final_result>(\{.*\})", failed)
#         if match:
#             try:
#                 result_json = json.loads(match.group(1))
#                 params = result_json.get("parameters", [])
#                 # Post-process each param
#                 for p in params:
#                     # parse string reference_range
#                     rr = p.get("reference_range")
#                     if isinstance(rr, str):
#                         try:
#                             p["reference_range"] = json.loads(rr)
#                         except json.JSONDecodeError:
#                             p["reference_range"] = ast.literal_eval(rr)
#                     # default null fields
#                     p["assessment"] = p.get("assessment") or ""
#                     p["unit"] = p.get("unit") or ""
#                 return params
#             except Exception as parse_exc:
#                 logger.error(f"Failed to parse failed_generation JSON: {parse_exc}")
#         # if we cannot recover
#         logger.exception("PydanticAI parsing error with failed_generation")
#         raise HTTPException(status_code=500, detail="Failed to parse LLM output.")

#     except LLMWhispererClientException as e:
#         logger.error(f"PydanticAI agent client error: {e}")
#         raise HTTPException(status_code=502, detail=str(e))

#     except Exception as e:
#         logger.exception("Unexpected parsing error")
#         raise HTTPException(
#             status_code=500,
#             detail="An unexpected error occurred while parsing the health report.",
#         )

# import logging
# from typing import List, Dict, Optional
# from fastapi import HTTPException
# from pydantic import BaseModel, Field
# from pydantic_ai import Agent
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
# from pydantic_ai.exceptions import ModelHTTPError
# from dotenv import load_dotenv

# load_dotenv()
# import json
# import re
# import ast


# # ─── SCHEMAS ────────────────────────────────────────────────────────────────────
# class LabParameter(BaseModel):
#     parameter: str = Field(..., description="Name of the lab test")
#     value: float = Field(..., description="Numeric result")
#     unit: Optional[str] = Field(None, description="Unit of measurement, if present")
#     reference_range: List[Optional[float]] = Field(
#         ..., description="Two-element list [low, high], with null for missing ends"
#     )
#     assessment: Optional[str] = Field(
#         None, description="One of 'low','normal','high', if provided"
#     )


# class LabReport(BaseModel):
#     parameters: List[LabParameter]


# # ─── AGENT SETUP ────────────────────────────────────────────────────────────────
# agent = Agent(
#     model="groq:llama-3.1-8b-instant",
#     system_prompt=(
#         "You are a strict healthcare assistant. You will ONLY extract lab test "
#         "parameters that are explicitly present in the medical report. "
#         "Output a JSON array of objects with keys [parameter, value, unit, reference_range, assessment]. "
#         "Use JSON null for missing values. Do NOT invent any parameters or ranges."
#     ),
#     result_type=LabReport,
#     temperature=0.0,
# )

# logger = logging.getLogger(__name__)


# # ─── PARSING FUNCTION ────────────────────────────────────────────────────────────
# def parse_health_report(raw_text: str) -> List[Dict]:
#     """
#     Sends the OCR’d text to PydanticAI and returns a clean list of parameter dicts.
#     Handles function-call failures by extracting fallback JSON and post-processing.
#     Raises HTTPException on unrecoverable errors.
#     """
#     try:
#         logger.debug(f"Sending to PydanticAI agent (text length: {len(raw_text)})")
#         run_result = agent.run_sync(raw_text)
#         report: LabReport = run_result.data
#         return [param.dict(exclude_none=True) for param in report.parameters]

#     except ModelHTTPError as e:
#         body = getattr(e, "body", {})
#         failed = body.get("error", {}).get("failed_generation", "")
#         match = re.search(r"final_result>(\{.*\})", failed)
#         if match:
#             try:
#                 result_json = json.loads(match.group(1))
#                 params = result_json.get("parameters", [])
#                 for p in params:
#                     # Normalize reference_range
#                     rr = p.get("reference_range")
#                     if isinstance(rr, str):
#                         try:
#                             p["reference_range"] = json.loads(rr)
#                         except Exception:
#                             p["reference_range"] = ast.literal_eval(rr)
#                     # Ensure proper types or remove
#                     if p.get("unit") is None:
#                         p.pop("unit", None)
#                     if p.get("assessment") is None:
#                         p.pop("assessment", None)
#                 return params
#             except Exception:
#                 logger.exception("Failed to parse fallback JSON")
#         logger.error("PydanticAI parsing error with failed_generation")
#         raise HTTPException(status_code=500, detail="Failed to parse LLM output.")

#     except LLMWhispererClientException as e:
#         logger.error(f"PydanticAI agent client error: {e}")
#         raise HTTPException(status_code=502, detail=str(e))

#     except Exception:
#         logger.exception("Unexpected parsing error")
#         raise HTTPException(
#             status_code=500,
#             detail="An unexpected error occurred while parsing the health report.",
#         )

# import logging
# from typing import List, Dict, Optional
# from fastapi import HTTPException
# from pydantic import BaseModel, Field
# from pydantic_ai import Agent
# from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
# from dotenv import load_dotenv

# load_dotenv()


# # ─── SCHEMAS ────────────────────────────────────────────────────────────────────
# class LabParameter(BaseModel):
#     parameter: str = Field(..., description="Name of the lab test")
#     value: float = Field(..., description="Numeric result")
#     unit: str = Field(..., description="Unit of measurement")
#     reference_range: List[Optional[float]] = Field(
#         ..., description="Two-element list [low, high], with null for missing ends"
#     )
#     assessment: str = Field(..., description="One of 'low','normal','high'")


# class LabReport(BaseModel):
#     parameters: List[LabParameter]


# # ─── AGENT SETUP ────────────────────────────────────────────────────────────────
# agent = Agent(
#     model="groq:llama-3.1-8b-instant",
#     system_prompt=(
#         "You are a strict healthcare assistant. You will ONLY extract lab test "
#         "parameters that are explicitly present in the medical report. "
#         "Output a JSON array of objects with keys [parameter, value, unit, reference_range, assessment]. "
#         "Use JSON null (not the word ‘none’) for any missing numeric values. "
#         "Do NOT invent any parameters or ranges."
#     ),
#     result_type=LabReport,
#     temperature=0.0,
# )

# logger = logging.getLogger(__name__)


# # ─── PARSING FUNCTION ────────────────────────────────────────────────────────────
# def parse_health_report(raw_text: str) -> List[Dict]:
#     """
#     Sends the OCR’d text to PydanticAI and returns a list of dicts.
#     Raises HTTPException on errors or schema mismatches.
#     """
#     try:
#         logger.debug(f"Sending to PydanticAI agent (text length: {len(raw_text)})")
#         run_result = agent.run_sync(raw_text)
#         report: LabReport = run_result.data
#         logger.debug(f"PydanticAI returned {len(report.parameters)} parameters.")
#         return [param.dict() for param in report.parameters]

#     except LLMWhispererClientException as e:
#         logger.error(f"PydanticAI agent client error: {e}")
#         raise HTTPException(status_code=502, detail=str(e))

#     except Exception as e:
#         logger.exception("PydanticAI parsing error")
#         raise HTTPException(
#             status_code=500,
#             detail=f"PydanticAI parsing error: {e}\nExcerpt: {raw_text[:200]}…",
#         )

import logging
from typing import List, Dict, Optional
from fastapi import HTTPException
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from unstract.llmwhisperer.client_v2 import LLMWhispererClientException
from dotenv import load_dotenv

load_dotenv()


# ─── SCHEMAS ────────────────────────────────────────────────────────────────────
class LabParameter(BaseModel):
    parameter: str = Field(..., description="Name of the lab test")
    value: float = Field(..., description="Numeric result")
    unit: str = Field(..., description="Unit of measurement")
    reference_range: List[Optional[float]] = Field(
        ..., description="Two-element list [low, high], with null for missing ends"
    )
    assessment: str = Field(..., description="One of 'low','normal','high'")


class LabReport(BaseModel):
    parameters: List[LabParameter]


# ─── AGENT SETUP ────────────────────────────────────────────────────────────────
agent = Agent(
    model="groq:llama-3.1-8b-instant",
    system_prompt=(
        "You are a strict healthcare assistant. You will ONLY extract lab test "
        "parameters that are explicitly present in the medical report. "
        "Output a JSON array of objects with keys [parameter, value, unit, reference_range, assessment]. "
        "Use JSON `null` (not the word ‘none’) for any missing numeric values. "
        "Do NOT invent any parameters or ranges."
    ),
    result_type=LabReport,
    temperature=0.0,
)

logger = logging.getLogger(__name__)


# ─── PARSING FUNCTION ────────────────────────────────────────────────────────────
def parse_health_report(raw_text: str) -> List[Dict]:
    """
    Sends the OCR’d text to PydanticAI and returns a list of dicts.
    Raises HTTPException on errors or schema mismatches.
    """
    try:
        logger.debug(f"Sending to PydanticAI agent (text length: {len(raw_text)})")
        run_result = agent.run_sync(raw_text)
        report: LabReport = run_result.data
        logger.debug(f"PydanticAI returned {len(report.parameters)} parameters.")
        return [param.dict() for param in report.parameters]

    except LLMWhispererClientException as e:
        logger.error(f"PydanticAI agent client error: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    except Exception as e:
        logger.exception("PydanticAI parsing error")
        raise HTTPException(
            status_code=500,
            detail=f"PydanticAI parsing error: {e}\nExcerpt: {raw_text[:200]}…",
        )
