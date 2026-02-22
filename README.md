# Attendance Sheet Digitization System

Convert handwritten college attendance sheets into structured Excel files using Gemini 2.5 Flash Vision AI — with optional Tesseract OCR fallback for offline use.

## Features

- 📸 Upload or capture attendance sheet images (camera or file upload)
- 🤖 AI-powered extraction using Gemini 2.5 Flash Vision
- 🔍 Fallback OCR via Tesseract for offline/local processing
- 📊 Structured Excel output matching original layout
- 🎯 Confidence scoring and validation with real-time accuracy metrics
- ✏️ Optional manual correction interface
- ⚠️ UNCLEAR cell highlighting in yellow
- 📦 Batch processing support for multiple sheets

## Setup

### 1. Clone / Download

```bash
git clone https://github.com/Yasaswini-ch/Automated-Attendance-Sheet-Digitization-System.git
cd Automated-Attendance-Sheet-Digitization-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### Install Tesseract OCR (optional, for offline fallback)

**Windows:**
1. Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install and add to system PATH

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 3. Set API Key

Create a `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

Or set as environment variable:
```bash
export GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Prepare Input Files

#### Students Database (`students.xlsx`)
- Excel file with student roll numbers
- Column header must contain "Roll" or "roll"
- One roll number per row (e.g., 323103282001, 323103282002, etc.)

#### Timetable (`3CSM1_Timetable.xlsx`)
- Columns: Period numbers (1–8) and corresponding staff names
- Mark lunch periods appropriately

#### Attendance Sheet Images
- Clear PNG, JPG, or JPEG images — recommended 300dpi or higher
- Ensure roll numbers and attendance marks are clearly visible

### 5. Run Locally

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
├── attendance_digitizer.py # CLI entry point / Tesseract OCR fallback
├── requirements.txt
└── README.md
```

## Architecture

1. **User uploads image** → Streamlit file uploader or camera
2. **Gemini 2.5 Flash** extracts structured JSON (deterministic, temperature=0)
3. **Second Gemini call** verifies extraction and scores confidence
4. **openpyxl** generates formatted Excel matching original layout
5. **Download button** serves the Excel file

> **Offline fallback:** Run `attendance_digitizer.py` directly to use Tesseract OCR without an API key.

## Output Excel Structure

- Row 1–2: College name and location (merged cells)
- Row 3–4: Day/Date, Department, Year/Semester, Section
- Row 5–6: Period headers
- Row 7+: Attendance data (one row per student)
- Bottom: Teacher names, subjects, signatures, totals

A secondary **Period-wise Sheet** is also generated with columns P1–P8 per student.

## Accuracy Metrics

The system provides real-time detection assessment:

- **OCR Confidence**: Average confidence score (0–100%)
- **Detection Accuracy**: Percentage of students successfully detected
- **Detection Status**:
  - 🟢 Good — 80%+ students detected
  - 🟡 Poor — 50–80% students detected
  - 🔴 Very Poor — <50% students detected

## CLI Usage

```bash
# Basic
python attendance_digitizer.py --students students.xlsx --timetable 3CSM1_Timetable.xlsx --image attendance.jpg --out output.xlsx

# With options
python attendance_digitizer.py \
    --students path/to/students.xlsx \
    --timetable path/to/timetable.xlsx \
    --image path/to/attendance.jpg \
    --out path/to/output.xlsx \
    --device cpu \
    --no-layoutlmv3

# Check dependencies
python attendance_digitizer.py --check
```

## Configuration

### Tesseract Path (if not in PATH)
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Custom Roll Number Pattern
```python
# Modify regex in the extraction function to match your institution's format
if re.match(r'^323103282\d{3}$', elem['text']):
```

## Notes

- UNCLEAR cells are highlighted yellow
- LUNCH column is highlighted in amber
- All roll numbers preserved exactly as written
- Teacher initials preserved exactly (e.g., Dr.M.B.S, P.M)
- Supported date formats: DD-MM-YY, DD/MM/YY, DD.MM.YY, YYYY-MM-DD

## Troubleshooting

**"Tesseract not found"** — Ensure Tesseract is installed and in your system PATH, or set the path manually (see Configuration above).

**Low detection accuracy** — Use high-resolution, well-lit images. Check that roll numbers match the expected pattern.

**Date not detected** — Ensure the date is printed clearly and follows a supported format.

**Memory issues** — Reduce image resolution, use `--device cpu`, or process in batches.

## Advanced: Hugging Face Models (Optional)

For enhanced table detection accuracy:

```bash
pip install torch transformers timm
```

Then run without the `--no-layoutlmv3` flag.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes and push
4. Open a Pull Request

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Google Gemini](https://deepmind.google/technologies/gemini/) for Vision AI extraction
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for offline OCR engine
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenCV](https://opencv.org/) for image preprocessing
- [Hugging Face](https://huggingface.co/) for advanced ML models
