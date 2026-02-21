import streamlit as st
import json
import os
from PIL import Image
import io
import base64
from dotenv import load_dotenv
from gemini_service import extract_attendance_data
from confidence_validator import validate_confidence
from excel_generator import generate_excel

# Load API key from .env file automatically
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

st.set_page_config(
    page_title="Attendance Sheet Digitization System",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Attendance Sheet Digitization System")
st.markdown("*Convert handwritten attendance sheets to structured Excel files using AI*")

# Session state
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "confidence_data" not in st.session_state:
    st.session_state.confidence_data = None
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

# --- Input Section ---
st.header("📤 Upload Attendance Sheet")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Image (JPG/PNG/PDF)",
        type=["jpg", "jpeg", "png", "pdf"],
        help="Upload a scanned attendance sheet"
    )

with col2:
    camera_image = st.camera_input("Or Capture with Camera")

# Determine image source
image_bytes = None
if uploaded_file:
    image_bytes = uploaded_file.read()
    st.session_state.image_bytes = image_bytes
elif camera_image:
    image_bytes = camera_image.read()
    st.session_state.image_bytes = image_bytes
elif st.session_state.image_bytes:
    image_bytes = st.session_state.image_bytes

# Image Preview
if image_bytes:
    st.subheader("🖼️ Image Preview")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption="Uploaded Attendance Sheet", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not preview image: {e}")

    # --- Process Button ---
    st.divider()

    if not GEMINI_API_KEY:
        st.error("⚠️ GEMINI_API_KEY not found. Please add it to your `.env` file:\n```\nGEMINI_API_KEY=your_key_here\n```")
    
    if st.button("🤖 Process with AI", type="primary", use_container_width=True, disabled=not GEMINI_API_KEY):
        with st.spinner("Extracting attendance data with Gemini Vision AI..."):
            try:
                extracted = extract_attendance_data(image_bytes, GEMINI_API_KEY)
                st.session_state.extracted_data = extracted
                st.success("✅ Extraction complete!")
            except Exception as e:
                st.error(f"❌ Extraction failed: {e}")

        if st.session_state.extracted_data:
            with st.spinner("Running confidence validation..."):
                try:
                    confidence = validate_confidence(image_bytes, st.session_state.extracted_data, GEMINI_API_KEY)
                    st.session_state.confidence_data = confidence
                except Exception as e:
                    st.warning(f"Confidence check failed: {e}")
                    st.session_state.confidence_data = None

# --- Results Section ---
if st.session_state.extracted_data:
    data = st.session_state.extracted_data

    st.divider()
    st.header("📊 Extracted Data")

    # Confidence Score
    if st.session_state.confidence_data:
        conf = st.session_state.confidence_data.get("confidence_analysis", {})
        overall = conf.get("overall_confidence_percent", "N/A")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            score = float(str(overall).replace("%","")) if str(overall).replace("%","").replace(".","").isdigit() else 0
            color = "🟢" if score >= 90 else ("🟡" if score >= 75 else "🔴")
            st.metric(f"{color} Confidence Score", f"{overall}%")
        with col2:
            st.metric("Total Rows Detected", conf.get("total_rows_detected", "N/A"))
        with col3:
            st.metric("Rows with UNCLEAR", conf.get("rows_with_unclear_values", "N/A"))

        if conf.get("notes"):
            st.info(f"📝 AI Notes: {conf['notes']}")
        if conf.get("suspected_mismatch_rows"):
            st.warning(f"⚠️ Suspected mismatches in rows: {conf['suspected_mismatch_rows']}")

    # Header Info
    st.subheader("📌 Header Information")
    header = data.get("header", {})
    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        st.write(f"**College:** {header.get('college_name', 'N/A')}")
        st.write(f"**Location:** {header.get('location', 'N/A')}")
    with hcol2:
        st.write(f"**Date:** {header.get('day_date', 'N/A')}")
        st.write(f"**Department:** {header.get('department_branch', 'N/A')}")
    with hcol3:
        st.write(f"**Year/Sem:** {header.get('year_semester', 'N/A')}")
        st.write(f"**Section:** {header.get('section', 'N/A')}")

    # Attendance Table Preview
    st.subheader("📋 Attendance Table Preview")
    attendance = data.get("attendance_table", [])
    if attendance:
        import pandas as pd
        df = pd.DataFrame(attendance)
        
        def highlight_unclear(val):
            if str(val).strip().upper() == "UNCLEAR":
                return "background-color: yellow; color: black"
            return ""
        
        styled_df = df.style.applymap(highlight_unclear)
        st.dataframe(styled_df, use_container_width=True, height=400)
        st.caption(f"Total rows: {len(attendance)}")
    else:
        st.warning("No attendance data extracted.")

    # Bottom Section
    st.subheader("👩‍🏫 Teacher & Subject Information")
    bottom = data.get("bottom_section", {})
    periods = ["period1","period2","period3","period4","period5","period6","period7","period8"]
    
    teacher_row = {"Field": "Teacher Name"}
    subject_row = {"Field": "Subject"}
    for p in periods:
        label = p.replace("period","Period ")
        teacher_row[label] = bottom.get("teacher_names", {}).get(p, "")
        subject_row[label] = bottom.get("subjects", {}).get(p, "")
    
    import pandas as pd
    bottom_df = pd.DataFrame([teacher_row, subject_row]).set_index("Field")
    st.dataframe(bottom_df, use_container_width=True)

    # Raw JSON toggle
    with st.expander("🔍 View Raw JSON"):
        st.json(data)

    # Optional editable table
    with st.expander("✏️ Manual Correction (Optional)"):
        st.info("Edit the JSON below and click 'Apply Changes' to update the data.")
        edited_json = st.text_area("Edit JSON:", value=json.dumps(data, indent=2), height=400)
        if st.button("Apply Changes"):
            try:
                st.session_state.extracted_data = json.loads(edited_json)
                st.success("Changes applied!")
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # Generate Excel
    st.divider()
    st.header("📥 Generate Excel File")
    
    if st.button("📊 Generate Excel", type="primary", use_container_width=True):
        with st.spinner("Generating Excel file..."):
            try:
                excel_bytes = generate_excel(st.session_state.extracted_data)
                st.success("✅ Excel file generated successfully!")
                
                st.download_button(
                    label="⬇️ Download Excel File",
                    data=excel_bytes,
                    file_name="attendance_sheet.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ Excel generation failed: {e}")
                st.exception(e)

else:
    st.info("👆 Upload an attendance sheet image and click 'Process with AI' to begin.")

# Footer
st.divider()
st.caption("Attendance Sheet Digitization System | Powered by Gemini 2.5 Flash Vision")
