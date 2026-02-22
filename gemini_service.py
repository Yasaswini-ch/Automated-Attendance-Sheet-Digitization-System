import google.generativeai as genai
import json
import re
from PIL import Image
import io


EXTRACTION_PROMPT = """You are a deterministic document transcription engine for college attendance sheets.
Extract data EXACTLY as visible. Apply these STRICT normalization rules:

ATTENDANCE VALUE RULES (most important):
- "/" (forward slash), "\\" (backslash), tick marks, checkmarks, any present-mark symbol → convert to "P"
- "A" or any absence mark → convert to "A"
- Blank/empty cell → leave as ""
- "UNCLEAR" only if truly unreadable

TEACHER NAME RULES:
- Extract teacher names from the bottom "Name of the Teacher" row
- Convert full names to initials/acronym format: "P.Mouni" → "P.M", "Dr.M.B.Suresh" → "Dr.M.B.S"
- Keep existing acronyms as-is (e.g. "P.M", "Dr.M.B.S" stay unchanged)
- Preserve "Dr." prefix if present

GENERAL RULES:
- Do NOT modify roll numbers
- Preserve capitalization exactly
- Detect blank vertical separator between period 4 and 5 as LUNCH
- Extract ALL rows visible (both left and right halves of the sheet)
- Do not hallucinate missing rows
- Preserve exact row order

Return output STRICTLY in this JSON format with no extra text:

{
  "header": {
    "college_name": "",
    "location": "",
    "day_date": "",
    "department_branch": "",
    "year_semester": "",
    "section": ""
  },
  "attendance_table": [
    {
      "ht_no": "",
      "period1": "",
      "period2": "",
      "period3": "",
      "period4": "",
      "lunch": "LUNCH",
      "period5": "",
      "period6": "",
      "period7": "",
      "period8": ""
    }
  ],
  "bottom_section": {
    "teacher_names": {
      "period1": "",
      "period2": "",
      "period3": "",
      "period4": "",
      "period5": "",
      "period6": "",
      "period7": "",
      "period8": ""
    },
    "subjects": {
      "period1": "",
      "period2": "",
      "period3": "",
      "period4": "",
      "period5": "",
      "period6": "",
      "period7": "",
      "period8": ""
    }
  }
}"""


def _normalize_attendance(value: str) -> str:
    """Post-process: convert all present-symbols to P, keep A, blank stays blank."""
    if value is None:
        return ""
    v = str(value).strip()
    if v.upper() == "UNCLEAR":
        return "UNCLEAR"
    if v.upper() == "LUNCH":
        return "LUNCH"
    if v == "":
        return ""
    # Any slash/backslash/tick/checkmark variants → P
    present_symbols = {"/", "\\", "✓", "✔", "√", "P", "p", "|"}
    if v in present_symbols:
        return "P"
    if v.upper() == "A":
        return "A"
    # If it's some other non-empty symbol, treat as P (present mark)
    return "P"


def _acronym_name(name: str) -> str:
    """Convert teacher name to acronym. e.g. 'P.Mouni' → 'P.M', 'Dr.M.B.Suresh' → 'Dr.M.B.S'"""
    if not name or name.strip() == "":
        return ""
    name = name.strip()
    
    # Already in acronym form (all dots, short) - keep as is
    # Pattern: X.Y or X.Y.Z etc. where each segment is 1-3 chars
    if re.match(r'^(Dr\.)?([A-Z]\.)+[A-Z]\.?$', name):
        return name
    
    # Has Dr. prefix
    prefix = ""
    if name.lower().startswith("dr."):
        prefix = "Dr."
        name = name[3:]
    elif name.lower().startswith("dr "):
        prefix = "Dr."
        name = name[3:]

    # Split by dots or spaces
    parts = re.split(r'[.\s]+', name)
    acronym_parts = [p[0].upper() for p in parts if p]
    
    return prefix + ".".join(acronym_parts)


def extract_attendance_data(image_bytes: bytes, api_key: str) -> dict:
    """Extract attendance data from image using Gemini Vision."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    img = Image.open(io.BytesIO(image_bytes))

    response = model.generate_content(
        [EXTRACTION_PROMPT, img],
        generation_config=genai.types.GenerationConfig(temperature=0.0)
    )

    json_data = _parse_json_response(response.text.strip())

    # ── Post-process: normalize attendance values ──────────────────────────
    for row in json_data.get("attendance_table", []):
        for period in ["period1","period2","period3","period4","period5","period6","period7","period8"]:
            row[period] = _normalize_attendance(row.get(period, ""))
        # Ensure lunch is always "LUNCH"
        row["lunch"] = "LUNCH"

    # ── Post-process: normalize teacher name acronyms ─────────────────────
    teacher_names = json_data.get("bottom_section", {}).get("teacher_names", {})
    for period, name in teacher_names.items():
        teacher_names[period] = _acronym_name(name)

    return json_data


def _parse_json_response(text: str) -> dict:
    """Robustly parse JSON from Gemini response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```', r'(\{[\s\S]*\})']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from response:\n{text[:500]}")
