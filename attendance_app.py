import streamlit as st
import pandas as pd
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import io
import base64

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Page configuration
st.set_page_config(
    page_title="Attendance Tracker",
    page_icon="📋",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        font-size: 2.5rem !important;
    }
    .present {
        color: #28a745;
        font-weight: bold;
    }
    .absent {
        color: #dc3545;
        font-weight: bold;
    }
    .lunch {
        color: #ffc107;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load timetable data
@st.cache_data
def load_timetable():
    timetable = pd.read_excel("3CSM1_Timetable.xlsx")
    return timetable

# Load students data
@st.cache_data
def load_students():
    students = pd.read_excel("students.xlsx")
    return students

# Get subject for a specific day and period
def get_subject(timetable_df, day, period):
    day_row = timetable_df[timetable_df['DAY'].str.upper() == day.upper()]
    if day_row.empty:
        return "N/A"
    
    period_cols = [
        '8:40AM-9:30AM',      # Period 1
        '9:30AM-10:20AM',     # Period 2
        '10:20AM-11:10AM',    # Period 3
        '11:10AM-12:00PM',    # Period 4
        '12:00PM-12:50PM',    # Period 5 (Lunch)
        '12:50PM-1:40PM',     # Period 6
        '1:40PM-2:30PM',      # Period 7
        '2:30PM-3:20PM',      # Period 8
        '3:20PM-5:00PM'       # Period 9
    ]
    
    if period < 1 or period > len(period_cols):
        return "N/A"
    
    col = period_cols[period - 1]
    if col in day_row.columns:
        return str(day_row[col].values[0])
    return "N/A"

def extract_attendance_from_image(image, day_name):
    """Extract attendance from uploaded image"""
    
    # Convert PIL to OpenCV
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # OCR
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    
    elements = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 20:
            text = data['text'][i].strip()
            if text:
                elements.append({
                    'text': text,
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                })
    
    # Find date
    date = ""
    section = "I"  # Default section
    
    # Enhanced date extraction patterns
    date_patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # DD-MM-YY or DD/MM/YY
        r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}',  # YY-MM-DD or YYYY-MM-DD
        r'\d{1,2}\.\d{1,2}\.\d{2,4}',      # DD.MM.YY
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # Capture group for date
    ]
    
    # Also look for date-related keywords
    date_keywords = ['date', 'dt', 'attendance date', 'day']
    
    for elem in elements:
        text_lower = elem['text'].lower()
        
        # Check for date patterns
        for pattern in date_patterns:
            match = re.search(pattern, elem['text'])
            if match:
                date = match.group(1) if match.groups() else match.group(0)
                break
        
        # Check for date keywords nearby
        if any(keyword in text_lower for keyword in date_keywords):
            # Look for date in nearby elements
            for other_elem in elements:
                if (abs(other_elem['y'] - elem['y']) < 50 and 
                    abs(other_elem['x'] - elem['x']) < 200 and
                    other_elem != elem):
                    for pattern in date_patterns:
                        match = re.search(pattern, other_elem['text'])
                        if match:
                            date = match.group(1) if match.groups() else match.group(0)
                            break
        
        # Look for section
        if 'section' in text_lower or elem['text'] in ['I', 'II', 'III']:
            if len(elem['text']) <= 3:
                section = elem['text']
    
    # If no date found, try to extract from common positions
    if not date:
        # Look for elements that might be dates in top area of image
        top_elements = [elem for elem in elements if elem['y'] < 100]
        for elem in top_elements:
            for pattern in date_patterns:
                match = re.search(pattern, elem['text'])
                if match:
                    date = match.group(1) if match.groups() else match.group(0)
                    break
    
    # Find all hall ticket numbers
    students = []
    for elem in elements:
        if re.match(r'^323103282\d{3}$', elem['text']):
            students.append(elem)
    
    # Sort students by roll number (numerical order)
    students.sort(key=lambda x: int(x['text']))
    
    # Process each student
    records = []
    for student in students:
        y_center = student['y'] + student['h'] // 2
        x_after = student['x'] + student['w'] + 5
        
        # Find marks in same row
        marks = []
        for elem in elements:
            elem_y = elem['y'] + elem['h'] // 2
            if abs(elem_y - y_center) < 25 and elem['x'] > x_after:
                text = elem['text']
                marks.append((elem['x'], text))
        
        marks.sort(key=lambda m: m[0])
        mark_values = [m[1] for m in marks]
        
        # Build record with attendance logic:
        # 'A' = Absent, anything else (including empty) = Present
        record = {
            'Roll_No': student['text'],
            'Date': date,
            'Section': section,
            'Day': day_name
        }
        
        # Periods 1-4
        for i in range(4):
            mark = mark_values[i] if i < len(mark_values) else ''
            status = 'Absent' if mark == 'A' else 'Present'
            record[f'Period_{i+1}'] = status
        
        # Period 5 is always LUNCH
        record['Period_5'] = 'LUNCH'
        
        # Periods 6-8
        for i in range(4, 7):
            idx = i - 1
            mark = mark_values[idx] if idx < len(mark_values) else ''
            status = 'Absent' if mark == 'A' else 'Present'
            record[f'Period_{i+2}'] = status
        
        records.append(record)
    
    # Calculate accuracy metrics
    detected_texts = [i for i in range(len(data['text'])) if int(data['conf'][i]) > 20 and data['text'][i].strip()]
    total_confidence = sum(int(data['conf'][i]) for i in detected_texts)
    avg_confidence = total_confidence / len(detected_texts) if detected_texts else 0
    
    # Expected vs detected students
    expected_students = 66  # Total students in 3CSM1
    detection_accuracy = (len(students) / expected_students * 100) if expected_students > 0 else 0
    
    # Detection status
    detection_status = "Good" if len(students) >= expected_students * 0.8 else "Poor" if len(students) >= expected_students * 0.5 else "Very Poor"
    if not date:
        detection_status = "Date Not Detected"
    elif len(students) == 0:
        detection_status = "Not Detected Properly"
    
    accuracy_info = {
        'ocr_confidence': avg_confidence,
        'detection_accuracy': detection_accuracy,
        'students_detected': len(students),
        'expected_students': expected_students,
        'detection_status': detection_status,
        'date_detected': bool(date),
        'total_text_elements': len(detected_texts)
    }
    
    return records, date, section, accuracy_info

def main():
    st.title("📋 Attendance Tracker - 3CSM1")
    st.markdown("---")
    
    # Load data
    timetable_df = load_timetable()
    students_df = load_students()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Day selection
        day_options = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT']
        selected_day = st.selectbox("Select Day", day_options)
        
        st.markdown("---")
        st.header("📊 Quick Stats")
        total_students = len(students_df)
        st.metric("Total Students", total_students)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload Attendance", "📋 View Attendance", "📅 Timetable"])
    
    # Tab 1: Upload
    with tab1:
        st.header("Upload Attendance Sheet")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an attendance image",
                type=['png', 'jpg', 'jpeg']
            )
        
        with col2:
            st.info("""
            **Instructions:**
            1. Upload a clear image of the attendance sheet
            2. The system will automatically extract:
               - Roll numbers
               - Date
               - Section
               - Attendance marks
            3. 'A' = Absent, anything else = Present
            4. Period 5 is always marked as LUNCH
            """)
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col_img1, col_img2 = st.columns(2)
            
            with col_img1:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True)
            
            with col_img2:
                st.subheader("Processing...")
                with st.spinner("Extracting attendance data..."):
                    records, date, section, accuracy_info = extract_attendance_from_image(image, selected_day)
                
                if records:
                    # Show detection status with appropriate color
                    status_color = {
                        "Good": "green",
                        "Poor": "orange", 
                        "Very Poor": "red",
                        "Date Not Detected": "red",
                        "Not Detected Properly": "red"
                    }.get(accuracy_info['detection_status'], "gray")
                    
                    if accuracy_info['detection_status'] in ["Good"]:
                        st.success(f"✅ {accuracy_info['detection_status']} Detection - {len(records)} student records")
                    elif accuracy_info['detection_status'] in ["Poor"]:
                        st.warning(f"⚠️ {accuracy_info['detection_status']} Detection - {len(records)} student records")
                    else:
                        st.error(f"❌ {accuracy_info['detection_status']} - {len(records)} student records")
                    
                    # Show date with status
                    if accuracy_info['date_detected']:
                        st.info(f"📅 Date: {date}")
                    else:
                        st.error("📅 Date: Not Detected Properly")
                    
                    st.info(f"🏫 Section: {section}")
                    st.info(f"📆 Day: {selected_day}")
                    
                    # Display accuracy metrics
                    st.markdown("---")
                    st.subheader("🎯 Accuracy Metrics")
                    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                    
                    with col_a1:
                        st.metric("OCR Confidence", f"{accuracy_info['ocr_confidence']:.1f}%")
                    
                    with col_a2:
                        st.metric("Detection Accuracy", f"{accuracy_info['detection_accuracy']:.1f}%")
                    
                    with col_a3:
                        st.metric("Students Found", f"{accuracy_info['students_detected']}/{accuracy_info['expected_students']}")
                    
                    with col_a4:
                        st.metric("Text Elements", accuracy_info['total_text_elements'])
                    
                    # Store in session state
                    if 'attendance_data' not in st.session_state:
                        st.session_state.attendance_data = []
                    
                    st.session_state.attendance_data.extend(records)
                    st.session_state.last_uploaded = records
                else:
                    st.error("❌ No student records found in the image")
                    
                    # Show diagnostic information
                    st.markdown("---")
                    st.subheader("🔍 Detection Diagnostics")
                    col_d1, col_d2, col_d3 = st.columns(3)
                    
                    with col_d1:
                        st.metric("Detection Status", accuracy_info['detection_status'])
                    
                    with col_d2:
                        st.metric("Date Detected", "Yes" if accuracy_info['date_detected'] else "No")
                    
                    with col_d3:
                        st.metric("Text Elements Found", accuracy_info['total_text_elements'])
                    
                    st.info("💡 **Tips for better detection:**")
                    st.write("- Ensure the image is clear and well-lit")
                    st.write("- Make sure roll numbers are visible")
                    st.write("- Check that the date format is recognizable")
                    st.write("- Try cropping the image to focus on the attendance table")
    
    # Tab 2: View Attendance
    with tab2:
        st.header("Attendance Records")
        
        if 'attendance_data' in st.session_state and st.session_state.attendance_data:
            # Convert to DataFrame
            df = pd.DataFrame(st.session_state.attendance_data)
            
            # Remove duplicates (keep latest)
            df = df.drop_duplicates(subset=['Roll_No', 'Date'], keep='last')
            
            # Sort by roll number (numerical order)
            df = df.sort_values('Roll_No', key=lambda x: pd.to_numeric(x))
            
            # Filter options
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                filter_date = st.multiselect(
                    "Filter by Date",
                    options=df['Date'].unique(),
                    default=[]
                )
            
            with col_f2:
                filter_roll = st.multiselect(
                    "Filter by Roll No",
                    options=sorted(df['Roll_No'].unique()),
                    default=[]
                )
            
            with col_f3:
                filter_day = st.multiselect(
                    "Filter by Day",
                    options=df['Day'].unique(),
                    default=[]
                )
            
            # Apply filters
            filtered_df = df.copy()
            if filter_date:
                filtered_df = filtered_df[filtered_df['Date'].isin(filter_date)]
            if filter_roll:
                filtered_df = filtered_df[filtered_df['Roll_No'].isin(filter_roll)]
            if filter_day:
                filtered_df = filtered_df[filtered_df['Day'].isin(filter_day)]
            
            # Display table with styling
            st.subheader(f"Showing {len(filtered_df)} records")
            
            # Create styled dataframe
            def color_status(val):
                if val == 'Present':
                    return 'background-color: #d4edda; color: #155724'
                elif val == 'Absent':
                    return 'background-color: #f8d7da; color: #721c24'
                elif val == 'LUNCH':
                    return 'background-color: #fff3cd; color: #856404'
                return ''
            
            # Apply styling
            styled_df = filtered_df.style.applymap(
                color_status, 
                subset=['Period_1', 'Period_2', 'Period_3', 'Period_4', 
                        'Period_5', 'Period_6', 'Period_7', 'Period_8']
            )
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Summary statistics
            st.markdown("---")
            st.subheader("📊 Summary Statistics")
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                total_present = (filtered_df[['Period_1', 'Period_2', 'Period_3', 'Period_4',
                                               'Period_6', 'Period_7', 'Period_8']] == 'Present').sum().sum()
                st.metric("Total Present", total_present)
            
            with col_s2:
                total_absent = (filtered_df[['Period_1', 'Period_2', 'Period_3', 'Period_4',
                                              'Period_6', 'Period_7', 'Period_8']] == 'Absent').sum().sum()
                st.metric("Total Absent", total_absent)
            
            with col_s3:
                attendance_rate = (total_present / (total_present + total_absent) * 100) if (total_present + total_absent) > 0 else 0
                st.metric("Attendance Rate", f"{attendance_rate:.1f}%")
            
            with col_s4:
                st.metric("Unique Students", filtered_df['Roll_No'].nunique())
            
            # Download options
            st.markdown("---")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="attendance_records.csv",
                    mime="text/csv"
                )
            
            with col_d2:
                # Excel download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, sheet_name='Attendance', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download as Excel",
                    data=excel_data,
                    file_name="attendance_records.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Clear data button
            if st.button("🗑️ Clear All Data", type="secondary"):
                st.session_state.attendance_data = []
                st.rerun()
        
        else:
            st.info("📤 No attendance data yet. Upload an image in the 'Upload Attendance' tab.")
    
    # Tab 3: Timetable
    with tab3:
        st.header("📅 Class Timetable - 3CSM1")
        
        # Display timetable
        st.dataframe(timetable_df, use_container_width=True, hide_index=True)
        
        # Show subjects for selected day
        st.markdown("---")
        st.subheader(f"Subjects for {selected_day}")
        
        period_times = [
            ("8:40AM-9:30AM", "Period 1"),
            ("9:30AM-10:20AM", "Period 2"),
            ("10:20AM-11:10AM", "Period 3"),
            ("11:10AM-12:00PM", "Period 4"),
            ("12:00PM-12:50PM", "Period 5 (LUNCH)"),
            ("12:50PM-1:40PM", "Period 6"),
            ("1:40PM-2:30PM", "Period 7"),
            ("2:30PM-3:20PM", "Period 8"),
            ("3:20PM-5:00PM", "Period 9")
        ]
        
        day_row = timetable_df[timetable_df['DAY'].str.upper() == selected_day.upper()]
        
        if not day_row.empty:
            cols = st.columns(3)
            for i, (time_col, period_name) in enumerate(period_times):
                with cols[i % 3]:
                    subject = day_row[time_col].values[0] if time_col in day_row.columns else "N/A"
                    
                    if "LUNCH" in period_name:
                        st.markdown(f"""
                        <div style="background: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px;">
                            <b>{period_name}</b><br>
                            <span style="color: #856404;">🍽️ {subject}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: #e7f3ff; padding: 10px; border-radius: 5px; margin: 5px;">
                            <b>{period_name}</b><br>
                            <span style="color: #004085;">📚 {subject}</span>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
