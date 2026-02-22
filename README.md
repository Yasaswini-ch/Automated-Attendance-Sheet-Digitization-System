# Attendance Sheet Digitization System

Convert handwritten college attendance sheets into structured Excel files using Gemini 2.5 Flash Vision AI.

## Features

- 📸 Upload or capture attendance sheet images
- 🤖 AI-powered extraction using Gemini 2.5 Flash Vision
- 📊 Structured Excel output matching original layout
- 🎯 Confidence scoring and validation
- ✏️ Optional manual correction interface
- ⚠️ UNCLEAR cell highlighting in yellow

## Setup

### 1. Clone / Download

```bash
cd Automated-Attendance-Digitization-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Key

Create a `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

Or set environment variable:
```bash
export GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run Locally

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → Connect your GitHub repo
4. Set main file: `app.py`
5. Add secret: `GEMINI_API_KEY = "your_key_here"` in Settings → Secrets
6. Deploy!

## Project Structure

```
attendance_app/
├── app.py                  # Main Streamlit UI
├── gemini_service.py       # Gemini Vision extraction
├── confidence_validator.py # Second-pass confidence scoring
├── excel_generator.py      # Excel generation with openpyxl
├── requirements.txt
└── README.md
```

## Architecture

1. **User uploads image** → Streamlit file uploader or camera
2. **Gemini 2.5 Flash** extracts structured JSON (deterministic, temperature=0)
3. **Second Gemini call** verifies extraction and scores confidence
4. **openpyxl** generates formatted Excel matching original layout
5. **Download button** serves the Excel file

## Output Excel Structure

- Row 1-2: College name and location (merged cells)
- Row 3-4: Day/Date, Department, Year/Semester, Section
- Row 5-6: Period headers
- Row 7+: Attendance data (one row per student)
- Bottom: Teacher names, subjects, signatures, totals

## Notes

- UNCLEAR cells are highlighted yellow
- LUNCH column is highlighted in amber
- All roll numbers preserved exactly as written
- Teacher initials preserved exactly (e.g., Dr.M.B.S, P.M)
