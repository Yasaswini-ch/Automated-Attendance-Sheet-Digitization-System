import streamlit as st
import json
import os
from PIL import Image
import io
from gemini_service import extract_attendance_data
from confidence_validator import validate_confidence
from excel_generator import generate_excel

st.set_page_config(
    page_title="Attendance Sheet Digitization System",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Attendance Sheet Digitization System")
st.markdown("*Convert handwritten attendance sheets to structured Excel files using AI*")

# ── API Key: Streamlit Cloud secrets → local .env → sidebar input ────────────
# 1. Try Streamlit Cloud secrets (set via share.streamlit.io → Settings → Secrets)
# ✅ New - safely handles missing secrets.toml
_key_from_secrets = ""
try:
    _key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    _key_from_secrets = ""
# 2. Try local environment variable (for running locally with .env)
_key_from_env = os.environ.get("GEMINI_API_KEY", "")

# 3. Pre-fill sidebar if key found, otherwise ask user to enter it
_key_auto = _key_from_secrets or _key_from_env

with st.sidebar:
    st.header("🔑 API Configuration")
    if _key_auto:
        st.success("✅ API key loaded automatically.")
        api_key_input = _key_auto
    else:
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Enter your Gemini API key. Held in memory only, never saved."
        )
        st.caption("🔒 Key is held in session memory only, never written to any file.")
        st.markdown("[Get a free key →](https://aistudio.google.com/apikey)")

GEMINI_API_KEY = api_key_input.strip() if api_key_input else ""

# ── Session state ────────────────────────────────────────────────────────────
for key in ["extracted_data", "confidence_data", "image_bytes"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Upload / Camera ──────────────────────────────────────────────────────────
st.header("📤 Upload Attendance Sheet")

tab1, tab2 = st.tabs(["📁 Upload File", "📷 Camera (Mobile / Webcam)"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload Image (JPG/PNG/PDF)",
        type=["jpg", "jpeg", "png", "pdf"],
        help="Upload a scanned attendance sheet"
    )

with tab2:
    st.info("📱 On mobile, this opens your camera. Tap the flip icon to switch front/back camera.")
    camera_image = st.camera_input("Capture Attendance Sheet")

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

# ── Image Preview ─────────────────────────────────────────────────────────────
if image_bytes:
    st.subheader("🖼️ Image Preview")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption="Attendance Sheet", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not preview image: {e}")

    st.divider()

    if not GEMINI_API_KEY:
        st.warning("⬅️ Please enter your Gemini API Key in the sidebar to proceed.")

    process_btn = st.button(
        "🤖 Process with AI",
        type="primary",
        use_container_width=True,
        disabled=not GEMINI_API_KEY
    )

    if process_btn:
        st.session_state.extracted_data = None
        st.session_state.confidence_data = None

        with st.spinner("🔍 Extracting attendance data with Gemini Vision AI..."):
            try:
                extracted = extract_attendance_data(image_bytes, GEMINI_API_KEY)
                st.session_state.extracted_data = extracted
                st.success("✅ Extraction complete!")
            except Exception as e:
                st.error(f"❌ Extraction failed: {e}")

        if st.session_state.extracted_data:
            with st.spinner("🎯 Running confidence validation..."):
                try:
                    confidence = validate_confidence(
                        image_bytes, st.session_state.extracted_data, GEMINI_API_KEY
                    )
                    st.session_state.confidence_data = confidence
                except Exception as e:
                    st.warning(f"Confidence check failed (non-critical): {e}")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.extracted_data:
    data = st.session_state.extracted_data

    st.divider()
    st.header("📊 Extracted Data")

    # Confidence Score
    if st.session_state.confidence_data:
        conf = st.session_state.confidence_data.get("confidence_analysis", {})
        overall = conf.get("overall_confidence_percent", "N/A")
        try:
            score = float(str(overall))
            color = "🟢" if score >= 90 else ("🟡" if score >= 75 else "🔴")
        except Exception:
            score = 0
            color = "⚪"

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{color} Confidence Score", f"{overall}%")
        c2.metric("Total Rows Detected", conf.get("total_rows_detected", "N/A"))
        c3.metric("Rows with UNCLEAR", conf.get("rows_with_unclear_values", "N/A"))

        if conf.get("notes"):
            st.info(f"📝 AI Notes: {conf['notes']}")
        if conf.get("suspected_mismatch_rows"):
            st.warning(f"⚠️ Suspected mismatches: {conf['suspected_mismatch_rows']}")

    # Header Info
    st.subheader("📌 Header Information")
    header = data.get("header", {})
    h1, h2, h3 = st.columns(3)
    h1.write(f"**College:** {header.get('college_name','N/A')}")
    h1.write(f"**Location:** {header.get('location','N/A')}")
    h2.write(f"**Date:** {header.get('day_date','N/A')}")
    h2.write(f"**Department:** {header.get('department_branch','N/A')}")
    h3.write(f"**Year/Sem:** {header.get('year_semester','N/A')}")
    h3.write(f"**Section:** {header.get('section','N/A')}")

    # Attendance Table
    st.subheader("📋 Attendance Table Preview")
    attendance = data.get("attendance_table", [])
    if attendance:
        import pandas as pd
        df = pd.DataFrame(attendance)

        def highlight_unclear(val):
            return "background-color: yellow; color: black" if str(val).strip().upper() == "UNCLEAR" else ""

        try:
            styled_df = df.style.map(highlight_unclear)
        except AttributeError:
            styled_df = df.style.applymap(highlight_unclear)

        st.dataframe(styled_df, use_container_width=True, height=400)
        st.caption(f"Total rows detected: {len(attendance)}")
    else:
        st.warning("No attendance rows extracted.")

    # Teacher / Subject Info
    st.subheader("👩‍🏫 Teacher & Subject Information")
    bottom = data.get("bottom_section", {})
    import pandas as pd
    teacher_row = {"Field": "Teacher Acronym"}
    subject_row = {"Field": "Subject"}
    for i in range(1, 9):
        p = f"period{i}"
        label = f"Period {i}"
        teacher_row[label] = bottom.get("teacher_names", {}).get(p, "")
        subject_row[label] = bottom.get("subjects", {}).get(p, "")
    st.dataframe(pd.DataFrame([teacher_row, subject_row]).set_index("Field"), use_container_width=True)

    with st.expander("🔍 View Raw JSON"):
        st.json(data)

    with st.expander("✏️ Manual Correction (Optional)"):
        st.info("Edit the JSON and click 'Apply Changes' before generating Excel.")
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
        with st.spinner("Generating Excel..."):
            try:
                excel_bytes = generate_excel(st.session_state.extracted_data)
                st.success("✅ Excel file ready!")
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
    if not image_bytes:
        st.info("👆 Upload an attendance sheet image or use the camera to begin.")

st.divider()
st.caption("Attendance Sheet Digitization System | Powered by Gemini Vision AI")
