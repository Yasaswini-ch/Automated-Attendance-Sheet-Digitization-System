# Automated Attendance Sheet Digitization System

A sophisticated Python-based system for automatically digitizing attendance sheets using OCR technology, specifically designed for educational institutions.

## 🚀 Features

- **Automatic Roll Number Detection**: Extracts student roll numbers from attendance sheets
- **Attendance Status Classification**: Identifies present/absent status for each period
- **Date & Section Extraction**: Automatically extracts date and section information
- **Web Interface**: User-friendly Streamlit-based web application
- **Advanced OCR**: Uses Tesseract OCR with confidence scoring
- **Accuracy Metrics**: Real-time detection quality assessment
- **Data Export**: Export processed data to Excel format
- **Batch Processing**: Process multiple attendance sheets

## 📋 System Requirements

- Python 3.8 or higher
- Tesseract OCR Engine
- Windows/Linux/macOS

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Yasaswini-ch/Automated-Attendance-Sheet-Digitization-System.git
cd Automated-Attendance-Sheet-Digitization-System
```

### 2. Install Tesseract OCR

**Windows:**
1. Download Tesseract from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install and note the installation path (usually: `C:\Program Files\Tesseract-OCR\`)
3. Add the installation directory to your system PATH

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 3. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## 📁 Required Input Files

Before running the system, prepare the following files:

### 1. Students Database (`students.xlsx`)
- Create an Excel file with student roll numbers
- Column header should contain "Roll" or "roll"
- Format: One roll number per row (e.g., 323103282001, 323103282002, etc.)

### 2. Timetable (`3CSM1_Timetable.xlsx`)
- Create an Excel file with period and staff information
- Columns: Period numbers (1-8) and corresponding staff names
- Mark lunch periods appropriately

### 3. Attendance Sheet Images
- Clear images of attendance sheets
- Supported formats: PNG, JPG, JPEG
- Ensure roll numbers and attendance marks are clearly visible
- Recommended resolution: 300dpi or higher

## 🚀 Execution Guide

### Method 1: Web Application (Recommended)

1. **Start the Streamlit App:**
```bash
streamlit run attendance_app.py
```

2. **Access the Application:**
- Open your browser and go to `http://localhost:8501`
- The web interface provides three main tabs:
  - **Upload Attendance**: Upload and process attendance images
  - **View Attendance**: Browse and filter processed records
  - **Timetable**: View class timetable information

3. **Processing Steps:**
   - Select the day of attendance
   - Upload the attendance sheet image
   - Review the accuracy metrics
   - Download the processed data

### Method 2: Command Line Interface

1. **Basic Processing:**
```bash
python attendance_digitizer.py --students "students.xlsx" --timetable "3CSM1_Timetable.xlsx" --image "attendance.jpg" --out "output.xlsx"
```

2. **With Custom Parameters:**
```bash
python attendance_digitizer.py \
    --students "path/to/students.xlsx" \
    --timetable "path/to/timetable.xlsx" \
    --image "path/to/attendance.jpg" \
    --out "path/to/output.xlsx" \
    --device "cpu" \
    --no-layoutlmv3
```

3. **Check Dependencies:**
```bash
python attendance_digitizer.py --check
```

## 📊 Output Format

The system generates Excel files with the following structure:

### Detailed Sheet
- Roll_No: Student roll number
- Date: Attendance date
- Section: Class section
- Day: Day of week
- Period_1 to Period_8: Attendance status for each period
- Staff: Teacher name for each period
- Status: Present/Absent/Lunch

### Period-wise Sheet
- Roll_No: Student roll number
- P1 to P8: Consolidated attendance status

## 🎯 Accuracy Metrics

The system provides real-time accuracy assessment:

- **OCR Confidence**: Average confidence score of text detection (0-100%)
- **Detection Accuracy**: Percentage of students successfully detected
- **Students Found**: Number of detected students vs expected total
- **Detection Status**: 
  - 🟢 Good: 80%+ students detected
  - 🟡 Poor: 50-80% students detected
  - 🔴 Very Poor: <50% students detected

## 🔧 Configuration

### Tesseract Path Configuration
If Tesseract is not in PATH, modify the path in `attendance_app.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Custom Roll Number Pattern
Update the roll number regex pattern in the extraction function:
```python
# Current pattern for 3CSM1 roll numbers
if re.match(r'^323103282\d{3}$', elem['text']):
```

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **"Tesseract not found" Error**
   - Ensure Tesseract OCR is properly installed
   - Check if Tesseract path is correctly configured
   - Verify Tesseract is in system PATH

2. **Low Detection Accuracy**
   - Use high-resolution, clear images
   - Ensure proper lighting and focus
   - Check if roll numbers follow the expected pattern
   - Try image preprocessing (contrast enhancement, noise reduction)

3. **Date Not Detected**
   - Verify date format in the attendance sheet
   - Supported formats: DD-MM-YY, DD/MM/YY, DD.MM.YY, YYYY-MM-DD
   - Ensure date is clearly visible and not handwritten

4. **Memory Issues**
   - Reduce image resolution before processing
   - Use CPU processing instead of GPU if memory is limited
   - Process images in batches for large datasets

### Performance Optimization

- **For better speed**: Use `--device "cuda"` if you have a compatible GPU
- **For better accuracy**: Use higher resolution images and ensure proper lighting
- **For large datasets**: Process images in batches and consider using the CLI

## 📈 Advanced Features

### Using Hugging Face Models (Optional)
The system supports advanced table detection using Hugging Face transformers:

```bash
pip install torch transformers timm
```

Then run without the `--no-layoutlmv3` flag for enhanced accuracy.

### Batch Processing
Process multiple images using a script:
```python
import os
from attendance_digitizer import main

image_folder = "path/to/images/"
for image_file in os.listdir(image_folder):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        # Process each image
        pass
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for OCR engine
- [Streamlit](https://streamlit.io/) for web interface framework
- [OpenCV](https://opencv.org/) for image processing
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Hugging Face](https://huggingface.co/) for advanced ML models

## 📞 Support

For support and queries:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the accuracy metrics for detection quality assessment

---

**Note**: This system is specifically optimized for the attendance format used in the case study but can be adapted for different formats by modifying the extraction patterns.
