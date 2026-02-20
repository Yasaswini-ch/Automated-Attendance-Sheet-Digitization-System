"""
Streamlit Attendance OCR App
Dark luxury aesthetic with gold accents
"""

import streamlit as st
import pandas as pd
import json
import logging
from pathlib import Path
import sys
import time
import tempfile
import os

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ocr_module import process_attendance, export_attendance_files
except ImportError as e:
    st.error(f"Failed to import OCR module: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Attendance OCR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark luxury theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');

/* Global styles */
.stApp {
    background: #09090f;
    color: #ffffff;
    font-family: 'DM Sans', sans-serif;
}

/* Header fonts */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Cormorant Garamond', serif;
    color: #c8a456;
    font-weight: 600;
}

/* Code blocks */
.stCode {
    font-family: 'DM Mono', monospace;
    background: #0f0f18;
    border: 1px solid #1a1a2e;
    border-radius: 8px;
    padding: 16px;
}

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Sidebar */
.css-1d391kg {
    background: #0f0f18;
    border-right: 1px solid #1a1a2e;
}

/* File uploader */
.stFileUploader {
    background: #0f0f18;
    border: 2px dashed #1a1a2e;
    border-radius: 12px;
    padding: 2rem;
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: #c8a456;
    background: #151520;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #c8a456, #b8941f);
    color: #09090f;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    padding: 0.75rem 2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(200, 164, 86, 0.2);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #d4b062, #c4a02f);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(200, 164, 86, 0.3);
}

.stButton > button:disabled {
    background: #1a1a2e;
    color: #666;
    transform: none;
    box-shadow: none;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f18;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
    padding: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #888;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: #c8a456;
    color: #09090f;
}

/* Dataframes */
.dataframe {
    background: #0f0f18;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
    overflow: hidden;
}

.dataframe th {
    background: #1a1a2e;
    color: #c8a456;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    padding: 1rem;
}

.dataframe td {
    background: #0f0f18;
    color: #ffffff;
    font-family: 'DM Mono', monospace;
    padding: 0.75rem 1rem;
    border-top: 1px solid #1a1a2e;
}

/* Metrics/KPI cards */
.metric-container {
    background: linear-gradient(135deg, #0f0f18, #151520);
    border: 1px solid #1a1a2e;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-container:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(200, 164, 86, 0.1);
    border-color: #c8a456;
}

.metric-label {
    color: #888;
    font-size: 0.875rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    color: #c8a456;
    font-size: 2.5rem;
    font-family: 'Cormorant Garamond', serif;
    font-weight: 700;
    margin-top: 0.5rem;
}

/* Chips */
.chip {
    display: inline-block;
    padding: 0.375rem 0.875rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 0.25rem;
    font-family: 'DM Sans', sans-serif;
}

.chip-absent {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.chip-teacher {
    background: rgba(200, 164, 86, 0.2);
    color: #c8a456;
    border: 1px solid rgba(200, 164, 86, 0.3);
}

/* Log messages */
.log-success {
    color: #10b981;
    font-family: 'DM Mono', monospace;
}

.log-warning {
    color: #f59e0b;
    font-family: 'DM Mono', monospace;
}

.log-error {
    color: #ef4444;
    font-family: 'DM Mono', monospace;
}

/* Meta pills */
.meta-pill {
    background: #1a1a2e;
    color: #c8a456;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
    margin: 0.25rem;
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
}

/* Download buttons */
.download-btn {
    background: linear-gradient(135deg, #1a1a2e, #252538);
    color: #c8a456;
    border: 1px solid #c8a456;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    font-family: 'DM Sans', sans-serif;
    text-decoration: none;
    display: inline-block;
    transition: all 0.3s ease;
}

.download-btn:hover {
    background: linear-gradient(135deg, #252538, #303045);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

def add_log(message, level="info"):
    """Add message to processing log"""
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    st.session_state.processing_log.append({
        "time": timestamp,
        "message": message,
        "level": level
    })

def render_log():
    """Render processing log with color coding"""
    if not st.session_state.processing_log:
        return
    
    st.markdown("### 📋 Processing Log")
    log_container = st.container()
    with log_container:
        for entry in st.session_state.processing_log[-10:]:  # Show last 10 entries
            icon = {"info": "✓", "warning": "⚠", "error": "✗"}.get(entry["level"], "•")
            css_class = {"info": "log-success", "warning": "log-warning", "error": "log-error"}.get(entry["level"], "")
            st.markdown(
                f'<span class="{css_class}">{icon} [{entry["time"]}] {entry["message"]}</span>',
                unsafe_allow_html=True
            )

def render_kpi_cards(data):
    """Render KPI cards"""
    if not data:
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_students = len(data.get("all_rolls", []))
    absent_count = len(data.get("absentees", []))
    present_count = total_students - absent_count
    attendance_pct = round((present_count / total_students * 100), 1) if total_students > 0 else 0
    teachers_count = len(data.get("teachers", []))
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{total_students}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Present</div>
            <div class="metric-value">{present_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Absent</div>
            <div class="metric-value">{absent_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Attendance %</div>
            <div class="metric-value">{attendance_pct}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Teachers</div>
            <div class="metric-value">{teachers_count}</div>
        </div>
        """, unsafe_allow_html=True)

def render_meta_pills(data):
    """Render metadata pills"""
    if not data:
        return
    
    date = data.get("date", "N/A")
    section = data.get("section", "N/A")
    periods = len(data.get("roll_details", [{}])[0]) - 2 if data.get("roll_details") else 0  # Subtract roll and absent columns
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <span class="meta-pill">📅 {date}</span>
        <span class="meta-pill">📚 {section}</span>
        <span class="meta-pill">⏰ {periods} Periods</span>
    </div>
    """, unsafe_allow_html=True)

def process_file(uploaded_file):
    """Process uploaded file"""
    try:
        # Clear previous log
        st.session_state.processing_log = []
        
        # Save uploaded file temporarily
        temp_path = Path(f"temp_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        add_log(f"Processing file: {uploaded_file.name}")
        add_log(f"File size: {uploaded_file.size / 1024:.1f} KB")
        
        # Process attendance
        result = process_attendance(str(temp_path), export_excel=True)
        
        add_log(f"✓ Successfully processed attendance data")
        add_log(f"  - Found {len(result.get('all_rolls', []))} students")
        add_log(f"  - {len(result.get('absentees', []))} absent")
        add_log(f"  - {len(result.get('teachers', []))} teachers")
        
        # Export separate files
        if result.get("all_rolls"):
            base_path = str(temp_path.parent / temp_path.stem)
            export_attendance_files(result, base_path)
            add_log("✓ Exported separate absentees/presents files")
        
        # Clean up
        temp_path.unlink()
        
        st.session_state.processed_data = result
        return True
        
    except Exception as e:
        add_log(f"✗ Error: {str(e)}", "error")
        logger.error(f"Processing error: {e}", exc_info=True)
        return False

# Main app
def main():
    # Hero header
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem; font-weight: 700;">Attendance OCR</h1>
        <p style="color: #888; font-size: 1.25rem; font-family: 'DM Sans', sans-serif;">
            Extract attendance data from sheets with AI-powered OCR
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two-column layout
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        # File upload section
        st.markdown("### 📤 Upload File")
        uploaded_file = st.file_uploader(
            "Choose attendance file",
            type=["jpg", "jpeg", "png", "pdf", "xlsx", "xls"],
            label_visibility="collapsed"
        )
        
        # Show image preview if uploaded
        if uploaded_file and uploaded_file.type.startswith('image/'):
            st.markdown("### 📷 Image Preview")
            st.image(uploaded_file, width='stretch', caption="Uploaded attendance sheet")
        
        # Process button
        process_disabled = uploaded_file is None
        if st.button("🔍 Extract Attendance", disabled=process_disabled, width='stretch'):
            if uploaded_file:
                with st.spinner("Processing attendance data..."):
                    success = process_file(uploaded_file)
                    if success:
                        st.rerun()
        
        # Processing log
        render_log()
    
    with col_right:
        # Results section
        if st.session_state.processed_data:
            data = st.session_state.processed_data
            
            # KPI cards
            render_kpi_cards(data)
            
            # Meta pills
            render_meta_pills(data)
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Attendance Grid", "🚫 Absentees", "👨‍🏫 Teachers", "📄 JSON"])
            
            with tab1:
                st.markdown("### Attendance Details")
                if data.get("roll_details"):
                    df = pd.DataFrame(data["roll_details"])
                    # Rename columns for better display
                    df = df.rename(columns={"roll": "Roll Number", "absent": "Status"})
                    
                    # Style the dataframe
                    def highlight_absent(val):
                        if val == "Absent":
                            return 'background-color: rgba(239, 68, 68, 0.2); color: #ef4444;'
                        elif val == "Present":
                            return 'background-color: rgba(16, 185, 129, 0.2); color: #10b981;'
                        elif val == "Lunch":
                            return 'background-color: rgba(245, 158, 11, 0.2); color: #f59e0b;'
                        return ''
                    
                    styled_df = df.style.applymap(highlight_absent, subset=['Status'])
                    for col in df.columns:
                        if col.startswith('P'):
                            styled_df = styled_df.applymap(highlight_absent, subset=[col])
                    
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.info("No attendance data available")
            
            with tab2:
                st.markdown("### Absent Students")
                absentees = data.get("absentees", [])
                if absentees:
                    # Show as chips
                    chips_html = ""
                    for roll in absentees:
                        chips_html += f'<span class="chip chip-absent">{roll}</span>'
                    st.markdown(chips_html, unsafe_allow_html=True)
                    
                    # Show as dataframe
                    df_absent = pd.DataFrame({"Roll Number": absentees})
                    st.dataframe(df_absent, use_container_width=True)
                else:
                    st.success("✓ No absent students!")
            
            with tab3:
                st.markdown("### Teachers")
                teachers = data.get("teachers", [])
                if teachers:
                    # Show as chips with period labels
                    schedule = data.get("schedule", {})
                    staff_per_period = schedule.get("staff_per_period", {})
                    
                    chips_html = ""
                    for period, period_teachers in staff_per_period.items():
                        for teacher in period_teachers:
                            chips_html += f'<span class="chip chip-teacher">{period}: {teacher}</span>'
                    
                    st.markdown(chips_html, unsafe_allow_html=True)
                    
                    # Show as dataframe
                    df_teachers = pd.DataFrame({"Teacher": teachers})
                    st.dataframe(df_teachers, use_container_width=True)
                else:
                    st.info("No teacher data available")
            
            with tab4:
                st.markdown("### Raw JSON Data")
                # Prepare clean JSON for display
                json_data = {
                    "date": data.get("date"),
                    "section": data.get("section"),
                    "total_students": len(data.get("all_rolls", [])),
                    "present": len(data.get("all_rolls", [])) - len(data.get("absentees", [])),
                    "absent": len(data.get("absentees", [])),
                    "attendance_percentage": round((len(data.get("all_rolls", [])) - len(data.get("absentees", []))) / len(data.get("all_rolls", [])) * 100, 1) if data.get("all_rolls") else 0,
                    "teachers": data.get("teachers", []),
                    "absentees": data.get("absentees", []),
                    "period_absentees": data.get("period_absentees", {})
                }
                
                st.json(json_data)
            
            # Download buttons
            st.markdown("### 📥 Downloads")
            col_download1, col_download2, col_download3 = st.columns(3)
            
            with col_download1:
                if data.get("excel_path"):
                    with open(data["excel_path"], "rb") as f:
                        st.download_button(
                            label="📊 Download Excel Report",
                            data=f.read(),
                            file_name=Path(data["excel_path"]).name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            with col_download2:
                json_str = json.dumps(json_data, indent=2)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"attendance_{data.get('date', 'unknown')}.json",
                    mime="application/json"
                )
            
            with col_download3:
                # Create CSV for absentees
                if data.get("absentees"):
                    df_absent = pd.DataFrame({"Roll Number": data["absentees"]})
                    csv = df_absent.to_csv(index=False)
                    st.download_button(
                        label="🚫 Download Absentees CSV",
                        data=csv,
                        file_name=f"absentees_{data.get('date', 'unknown')}.csv",
                        mime="text/csv"
                    )
        
        else:
            # Placeholder when no data
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background: #0f0f18; border-radius: 16px; border: 1px solid #1a1a2e;">
                <h3 style="color: #888; margin-bottom: 1rem;">No Data Yet</h3>
                <p style="color: #666; font-family: 'DM Sans', sans-serif;">
                    Upload an attendance file and click "Extract Attendance" to get started
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
