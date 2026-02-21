import google.generativeai as genai
import json
import re
from PIL import Image
import io


CONFIDENCE_PROMPT = """You are a quality assurance engine for attendance sheet extraction.

Compare the extracted JSON data against the original attendance sheet image.

Apply these deductions:
- Deduct 2% per UNCLEAR field found
- Deduct 3% per suspected mismatch between extracted data and image
- Never return 100% unless no issues found at all
- Be specific about which rows have issues

Return STRICTLY this JSON with no additional text:

{
  "confidence_analysis": {
    "overall_confidence_percent": 0,
    "total_rows_detected": 0,
    "rows_with_unclear_values": 0,
    "suspected_mismatch_rows": [],
    "notes": ""
  }
}"""


def validate_confidence(image_bytes: bytes, extracted_data: dict, api_key: str) -> dict:
    """Validate extraction confidence using second Gemini call."""
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    img = Image.open(io.BytesIO(image_bytes))
    
    prompt_with_data = f"""{CONFIDENCE_PROMPT}

Extracted JSON to verify:
{json.dumps(extracted_data, indent=2)}
"""
    
    response = model.generate_content(
        [prompt_with_data, img],
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
        )
    )
    
    raw_text = response.text.strip()
    return _parse_json_response(raw_text)


def _parse_json_response(text: str) -> dict:
    """Robustly parse JSON from response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
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
    
    # Return default if parsing fails
    return {
        "confidence_analysis": {
            "overall_confidence_percent": "N/A",
            "total_rows_detected": "N/A",
            "rows_with_unclear_values": "N/A",
            "suspected_mismatch_rows": [],
            "notes": "Could not parse confidence response"
        }
    }
