import google.generativeai as genai
import json
import re
import base64
from PIL import Image
import io


EXTRACTION_PROMPT = """You are a deterministic document transcription engine.
Extract data EXACTLY as visible from this attendance sheet image.

STRICT RULES:
- Do NOT guess unclear characters. If unreadable, write "UNCLEAR".
- Do NOT modify roll numbers.
- Preserve capitalization exactly.
- Only allowed attendance values: "A", "/", "P", or blank "".
- The blank vertical separator between period 4 and 5 is "LUNCH".
- Preserve row order exactly.
- If lab spans 3 continuous periods, note them as same subject.
- Do not hallucinate missing rows.
- Extract ALL rows visible, both left and right columns of the sheet.

Return output STRICTLY in this JSON format with no additional text:

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


def extract_attendance_data(image_bytes: bytes, api_key: str) -> dict:
    """Extract attendance data from image using Gemini Vision."""
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Convert bytes to PIL Image for Gemini
    img = Image.open(io.BytesIO(image_bytes))
    
    response = model.generate_content(
        [EXTRACTION_PROMPT, img],
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,  # Deterministic
        )
    )
    
    raw_text = response.text.strip()
    
    # Extract JSON from response
    json_data = _parse_json_response(raw_text)
    
    return json_data


def _parse_json_response(text: str) -> dict:
    """Robustly parse JSON from Gemini response."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code block
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'(\{[\s\S]*\})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    raise ValueError(f"Could not parse JSON from Gemini response. Raw response:\n{text[:500]}")
