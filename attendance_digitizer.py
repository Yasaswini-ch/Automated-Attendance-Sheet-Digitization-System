import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


DEFAULT_STUDENTS_FILE = "students.xlsx"
DEFAULT_TIMETABLE_FILE = "3CSM1_Timetable.xlsx"
DEFAULT_ATTENDANCE_IMAGE = "attendance.jpg"
DEFAULT_OUTPUT_FILE = "final_attendance.xlsx"

TABLE_DET_MODEL = "microsoft/table-transformer-detection"
TABLE_STRUCT_MODEL = "microsoft/table-transformer-structure-recognition"
TROCR_MODEL = "microsoft/trocr-base-handwritten"
LAYOUTLMV3_MODEL = "microsoft/layoutlmv3-base-finetuned-funsd"


@dataclass
class StructureGrid:
    table_bbox: Tuple[int, int, int, int]
    row_boxes: List[Tuple[int, int, int, int]]
    col_boxes: List[Tuple[int, int, int, int]]


def normalize_period(value) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


def parse_timetable(path: str) -> Tuple[Dict[int, str], Set[int]]:
    import pandas as pd

    df = pd.read_excel(path)
    if df.empty:
        return {}, set()

    normalized_cols = {c.lower().strip(): c for c in df.columns}
    period_col = next((normalized_cols[c] for c in normalized_cols if "period" in c), None)
    staff_col = next((normalized_cols[c] for c in normalized_cols if "staff" in c or "faculty" in c), None)
    lunch_col = next((normalized_cols[c] for c in normalized_cols if "lunch" in c), None)

    period_to_staff: Dict[int, str] = {}
    lunch_periods: Set[int] = set()

    if period_col is not None:
        chosen_staff_col = staff_col if staff_col is not None else df.columns[-1]
        for _, row in df.iterrows():
            period = normalize_period(row.get(period_col))
            if period is None:
                continue
            staff = str(row.get(chosen_staff_col, "")).strip()
            period_to_staff[period] = staff

            lunch_flag = str(row.get(lunch_col, "")).strip().lower() if lunch_col else ""
            if lunch_flag in {"1", "true", "yes", "y", "l", "lunch"}:
                lunch_periods.add(period)
            elif "lunch" in staff.lower() or staff.lower() == "l":
                lunch_periods.add(period)

        if period_to_staff:
            return period_to_staff, lunch_periods

    for col in df.columns:
        period = normalize_period(col)
        if period is None:
            continue
        series = df[col].dropna().astype(str).str.strip()
        staff = ""
        if not series.empty:
            non_lunch = [v for v in series.tolist() if "lunch" not in v.lower() and v.lower() != "l"]
            staff = non_lunch[0] if non_lunch else series.iloc[0]
        period_to_staff[period] = staff
        if "lunch" in str(col).lower() or any("lunch" in v.lower() or v.lower() == "l" for v in series.tolist()):
            lunch_periods.add(period)

    return period_to_staff, lunch_periods


def load_students(path: str) -> List[str]:
    import pandas as pd

    df = pd.read_excel(path)
    if df.empty:
        return []
    roll_col = next((c for c in df.columns if "roll" in str(c).lower()), df.columns[0])
    return [str(v).strip().replace(" ", "") for v in df[roll_col].dropna().tolist()]


def load_hf_components(device: str = "cpu", use_layoutlmv3: bool = True):
    try:
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies for HF pipeline. Install with: "
            "pip install torch transformers pillow pandas openpyxl"
        ) from exc

    det_processor = AutoImageProcessor.from_pretrained(TABLE_DET_MODEL)
    det_model = AutoModelForObjectDetection.from_pretrained(TABLE_DET_MODEL).to(device)

    struct_processor = AutoImageProcessor.from_pretrained(TABLE_STRUCT_MODEL)
    struct_model = AutoModelForObjectDetection.from_pretrained(TABLE_STRUCT_MODEL).to(device)

    trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)

    result = {
        "torch": torch,
        "det_processor": det_processor,
        "det_model": det_model,
        "struct_processor": struct_processor,
        "struct_model": struct_model,
        "trocr_processor": trocr_processor,
        "trocr_model": trocr_model,
    }

    if use_layoutlmv3:
        layout_processor = LayoutLMv3Processor.from_pretrained(LAYOUTLMV3_MODEL)
        layout_model = LayoutLMv3ForTokenClassification.from_pretrained(LAYOUTLMV3_MODEL).to(device)
        result["layout_processor"] = layout_processor
        result["layout_model"] = layout_model

    return result


def _predict_boxes(image, processor, model, torch_module, threshold: float = 0.5):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(next(model.parameters()).device)

    with torch_module.no_grad():
        outputs = model(pixel_values=pixel_values)

    target_sizes = torch_module.tensor([image.size[::-1]], device=next(model.parameters()).device)
    processed = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    boxes = [[int(v) for v in box] for box in processed["boxes"].detach().cpu().tolist()]
    scores = [float(v) for v in processed["scores"].detach().cpu().tolist()]
    labels = [int(v) for v in processed["labels"].detach().cpu().tolist()]
    id2label = model.config.id2label

    return [
        {"label": id2label.get(lbl, str(lbl)).lower(), "score": scr, "bbox": tuple(bx)}
        for lbl, scr, bx in zip(labels, scores, boxes)
    ]


def detect_table_and_structure(image, hf) -> StructureGrid:
    det_predictions = _predict_boxes(image, hf["det_processor"], hf["det_model"], hf["torch"], threshold=0.4)
    table_candidates = [p for p in det_predictions if "table" in p["label"]]
    if not table_candidates:
        raise ValueError("No table detected by microsoft/table-transformer-detection.")

    table = sorted(table_candidates, key=lambda p: p["score"], reverse=True)[0]
    x1, y1, x2, y2 = table["bbox"]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.size[0], x2), min(image.size[1], y2)

    table_crop = image.crop((x1, y1, x2, y2))
    struct_predictions = _predict_boxes(
        table_crop, hf["struct_processor"], hf["struct_model"], hf["torch"], threshold=0.45
    )

    row_boxes, col_boxes = [], []
    for p in struct_predictions:
        bx1, by1, bx2, by2 = p["bbox"]
        gx1, gy1, gx2, gy2 = x1 + bx1, y1 + by1, x1 + bx2, y1 + by2
        if "row" in p["label"]:
            row_boxes.append((gx1, gy1, gx2, gy2))
        elif "column" in p["label"] or "col" in p["label"]:
            col_boxes.append((gx1, gy1, gx2, gy2))

    if len(row_boxes) < 2 or len(col_boxes) < 2:
        raise ValueError("Structure model did not detect enough rows/columns.")

    row_boxes = sorted(row_boxes, key=lambda b: (b[1] + b[3]) / 2)
    col_boxes = sorted(col_boxes, key=lambda b: (b[0] + b[2]) / 2)
    return StructureGrid((x1, y1, x2, y2), row_boxes, col_boxes)


def trocr_read(image, hf) -> str:
    pixel_values = hf["trocr_processor"](images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(next(hf["trocr_model"].parameters()).device)

    with hf["torch"].no_grad():
        generated_ids = hf["trocr_model"].generate(pixel_values)

    text = hf["trocr_processor"].batch_decode(generated_ids, skip_special_tokens=True)[0]
    return str(text).strip()


def refine_tokens_with_layoutlmv3(tokens: List[str], boxes: List[Tuple[int, int, int, int]], image, hf) -> List[str]:
    if not tokens:
        return tokens

    w, h = image.size
    norm_boxes = []
    for x1, y1, x2, y2 in boxes:
        norm_boxes.append(
            [
                int(max(0, min(1000, 1000 * x1 / max(w, 1)))),
                int(max(0, min(1000, 1000 * y1 / max(h, 1)))),
                int(max(0, min(1000, 1000 * x2 / max(w, 1)))),
                int(max(0, min(1000, 1000 * y2 / max(h, 1)))),
            ]
        )

    encoding = hf["layout_processor"](
        image, words=tokens, boxes=norm_boxes, return_tensors="pt", truncation=True, padding="max_length", max_length=512
    )
    encoding = {k: v.to(next(hf["layout_model"].parameters()).device) for k, v in encoding.items()}

    with hf["torch"].no_grad():
        outputs = hf["layout_model"](**encoding)

    pred_ids = outputs.logits.argmax(-1).detach().cpu().tolist()[0]
    labels = hf["layout_model"].config.id2label
    cleaned = []
    for token, pred in zip(tokens, pred_ids[1 : 1 + len(tokens)]):
        label = labels.get(int(pred), "O")
        if label in {"O", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"}:
            cleaned.append(token)
    return cleaned if cleaned else tokens


def normalize_roll(text: str) -> str:
    return re.sub(r"\s+", "", str(text).strip())


def classify_status(text: str, is_lunch: bool) -> str:
    if is_lunch:
        return "L"
    t = re.sub(r"\s+", "", str(text)).lower()
    return "Absent" if t in {"a", "abs", "absent"} else "Present"


def extract_cell_text(roll, period, ocr_results, grid):
    row_period_text, roll_to_row = ocr_results
    row_idx = roll_to_row.get(normalize_roll(roll))
    if row_idx is None:
        return ""
    return row_period_text.get((row_idx, int(period)), "").strip()


def build_cell_text_from_structure(image, grid: StructureGrid, hf, periods: List[int]):
    row_period_text: Dict[Tuple[int, int], str] = {}
    roll_to_row: Dict[str, int] = {}
    raw_tokens: List[str] = []
    raw_boxes: List[Tuple[int, int, int, int]] = []

    period_cols = grid.col_boxes[1 : 1 + len(periods)]
    data_rows = grid.row_boxes[1:]

    for row_idx, row_box in enumerate(data_rows, start=1):
        rx1, ry1, rx2, ry2 = row_box

        c0x1, c0y1, c0x2, c0y2 = grid.col_boxes[0]
        roll_crop = image.crop((max(rx1, c0x1), max(ry1, c0y1), min(rx2, c0x2), min(ry2, c0y2)))
        roll_text = normalize_roll(trocr_read(roll_crop, hf))
        if roll_text:
            roll_to_row[roll_text] = row_idx
            raw_tokens.append(roll_text)
            raw_boxes.append((max(rx1, c0x1), max(ry1, c0y1), min(rx2, c0x2), min(ry2, c0y2)))

        for col_idx, period in enumerate(periods):
            if col_idx >= len(period_cols):
                break
            px1, py1, px2, py2 = period_cols[col_idx]
            cell_box = (max(rx1, px1), max(ry1, py1), min(rx2, px2), min(ry2, py2))
            text = trocr_read(image.crop(cell_box), hf)
            row_period_text[(row_idx, int(period))] = text
            raw_tokens.append(text)
            raw_boxes.append(cell_box)

    return row_period_text, roll_to_row, raw_tokens, raw_boxes


def build_outputs(students, periods, period_to_staff, lunch_periods, grouped_ocr):
    import pandas as pd

    detailed_records, matrix_records = [], []
    for roll in students:
        row = {"Roll No": roll}
        for period in periods:
            cell_text = extract_cell_text(roll, period, grouped_ocr, grid=None)
            status = classify_status(cell_text, period in lunch_periods)
            detailed_records.append(
                {
                    "Roll No": roll,
                    "Period": period,
                    "Staff": period_to_staff.get(period, ""),
                    "Status": status,
                    "OCR Text": cell_text,
                }
            )
            row[f"P{period}"] = status
        matrix_records.append(row)

    return pd.DataFrame(detailed_records), pd.DataFrame(matrix_records)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attendance digitizer using Table Transformer + TrOCR + optional LayoutLMv3"
    )
    parser.add_argument("--students", default=DEFAULT_STUDENTS_FILE)
    parser.add_argument("--timetable", default=DEFAULT_TIMETABLE_FILE)
    parser.add_argument("--image", default=DEFAULT_ATTENDANCE_IMAGE)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--no-layoutlmv3", action="store_true")
    parser.add_argument("--check", action="store_true", help="Only validate paths/dependencies")
    return parser.parse_args()


def resolve_existing_path(path_value: str, label: str) -> Path:
    path = Path(path_value).expanduser()
    if path.exists():
        return path.resolve()
    cwd = Path.cwd().resolve()
    raise FileNotFoundError(
        f"{label} file not found: '{path_value}'. Checked from working directory: '{cwd}'. "
        "Use an absolute path or run command from the file directory."
    )


def resolve_output_path(path_value: str) -> Path:
    out_path = Path(path_value).expanduser()
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path.resolve()


def main():
    args = parse_args()

    students_path = resolve_existing_path(args.students, "Students")
    timetable_path = resolve_existing_path(args.timetable, "Timetable")
    image_path = resolve_existing_path(args.image, "Attendance image")
    output_path = resolve_output_path(args.out)

    if args.check:
        load_hf_components(device=args.device, use_layoutlmv3=not args.no_layoutlmv3)
        print("Check passed. Files and dependencies are available.")
        return

    from PIL import Image
    import pandas as pd

    students = load_students(str(students_path))
    if not students:
        raise ValueError("Students file has no rows.")

    period_to_staff, lunch_periods = parse_timetable(str(timetable_path))
    periods = sorted(period_to_staff.keys()) if period_to_staff else list(range(1, 9))

    image = Image.open(str(image_path)).convert("RGB")
    hf = load_hf_components(device=args.device, use_layoutlmv3=not args.no_layoutlmv3)

    structure_grid = detect_table_and_structure(image, hf)
    row_period_text, roll_to_row, raw_tokens, raw_boxes = build_cell_text_from_structure(image, structure_grid, hf, periods)

    if not args.no_layoutlmv3:
        _ = refine_tokens_with_layoutlmv3(raw_tokens, raw_boxes, image, hf)

    grouped_ocr = (row_period_text, roll_to_row)
    detailed_df, matrix_df = build_outputs(students, periods, period_to_staff, lunch_periods, grouped_ocr)

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        matrix_df.to_excel(writer, index=False, sheet_name="Period_Wise")
        detailed_df.to_excel(writer, index=False, sheet_name="Detailed")

    print(f"Saved attendance to {output_path}")
    print(matrix_df.head(10))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}")
