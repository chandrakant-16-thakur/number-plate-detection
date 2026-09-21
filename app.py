import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import sqlite3
import os
import re
import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Number Plate Detection",
    page_icon="🚗",
    layout="wide"
)

MODEL_PATH = "models/best.pt"
DB_PATH = "plates.db"

BLACKLIST = [
    "MH31AB1234",
    "HR98AA7777"
]


# =========================================================
# DATABASE
# =========================================================

def init_database():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT,
            image_name TEXT,
            date_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


init_database()


# =========================================================
# LOAD AI MODELS
# =========================================================

@st.cache_resource
def load_models():

    model = YOLO(MODEL_PATH)

    reader = easyocr.Reader(
        ['en'],
        gpu=False
    )

    return model, reader


# =========================================================
# OCR CORRECTION
# =========================================================

def digits_to_letters(segment):

    return (
        segment
        .replace("0", "O")
        .replace("1", "I")
        .replace("2", "Z")
        .replace("5", "S")
        .replace("6", "G")
        .replace("8", "B")
    )


def letters_to_digits(segment):

    return (
        segment
        .replace("O", "0")
        .replace("Q", "0")
        .replace("D", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
        .replace("G", "6")
        .replace("B", "8")
        .replace("Z", "2")
    )


def fix_indian_plate(plate):

    plate = plate.upper()

    plate = re.sub(
        r"[^A-Z0-9]",
        "",
        plate
    )

    # Remove IND prefix
    if plate.startswith("IND"):
        plate = plate[3:]

    # -----------------------------------------------------
    # BH SERIES
    # Example: 22BH6517A
    # -----------------------------------------------------

    if len(plate) >= 2 and "BH" in plate:

        bh_index = plate.find("BH")

        if bh_index == 2:

            prefix = letters_to_digits(
                plate[:2]
            )

            rest = plate[4:]

            number_part = "".join(
                c for c in rest
                if c.isdigit()
            )[:4]

            letter_part = "".join(
                c for c in rest
                if c.isalpha()
            )[:2]

            return (
                prefix +
                "BH" +
                number_part +
                letter_part
            )

    # -----------------------------------------------------
    # NORMAL INDIAN PLATE
    #
    # LL DD L-LL DDDD
    #
    # Example:
    # MH12DE1433
    # MH12E1433
    # -----------------------------------------------------

    if 9 <= len(plate) <= 11:

        state = digits_to_letters(
            plate[0:2]
        )

        rto = letters_to_digits(
            plate[2:4]
        )

        series = digits_to_letters(
            plate[4:-4]
        )

        number = letters_to_digits(
            plate[-4:]
        )

        return (
            state +
            rto +
            series +
            number
        )

    # Fallback
    if len(plate) >= 2:

        plate = (
            digits_to_letters(
                plate[:2]
            )
            + plate[2:]
        )

    return plate


# =========================================================
# PLATE VALIDATION
# =========================================================

PLATE_PATTERNS = [

    # Normal Indian plate
    r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$",

    # BH series
    r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"
]


def is_valid_plate(plate_number):

    plate_number = plate_number.upper().strip()

    return any(
        re.match(
            pattern,
            plate_number
        )
        for pattern in PLATE_PATTERNS
    )


# =========================================================
# OCR CLEANING
# =========================================================

def clean_ocr_text(
    text_items,
    min_confidence=0.50
):

    filtered = [
        item
        for item in text_items
        if item[2] >= min_confidence
    ]

    def box_position(item):

        box = item[0]

        xs = [
            point[0]
            for point in box
        ]

        ys = [
            point[1]
            for point in box
        ]

        return (
            min(ys),
            min(xs)
        )

    # Sort OCR text left-to-right
    filtered.sort(
        key=box_position
    )

    plate_number = "".join(
        item[1]
        for item in filtered
    )

    plate_number = re.sub(
        r"[^A-Z0-9]",
        "",
        plate_number.upper()
    )

    return fix_indian_plate(
        plate_number
    )


# =========================================================
# DATABASE FUNCTIONS
# =========================================================

def save_plate(
    plate_number,
    image_name
):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO plates
        (
            plate_number,
            image_name,
            date_time
        )
        VALUES (?, ?, ?)
        """,
        (
            plate_number,
            image_name,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    conn.commit()

    conn.close()


def get_history():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            plate_number,
            image_name,
            date_time
        FROM plates
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


# =========================================================
# NUMBER PLATE DETECTION
# =========================================================

def detect_plate(
    image,
    model,
    reader
):

    results = model.predict(
        image,
        conf=0.40,
        verbose=False
    )

    output_image = image.copy()

    detections = []

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(
                box.conf[0]
            )

            # Padding around YOLO crop
            pad_x = int(
                (x2 - x1) * 0.06
            )

            pad_y = int(
                (y2 - y1) * 0.15
            )

            px1 = max(
                0,
                x1 - pad_x
            )

            py1 = max(
                0,
                y1 - pad_y
            )

            px2 = min(
                image.shape[1],
                x2 + pad_x
            )

            py2 = min(
                image.shape[0],
                y2 + pad_y
            )

            plate = image[
                py1:py2,
                px1:px2
            ]

            if plate.size == 0:
                continue

            # ---------------------------------------------
            # PREPROCESSING
            # ---------------------------------------------

            gray = cv2.cvtColor(
                plate,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.resize(
                gray,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            candidates = []

            # Variant 1
            blurred = cv2.GaussianBlur(
                gray,
                (5, 5),
                0
            )

            _, thresh1 = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY +
                cv2.THRESH_OTSU
            )

            candidates.append(
                thresh1
            )

            # Variant 2
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            local_contrast = clahe.apply(
                gray
            )

            _, thresh2 = cv2.threshold(
                local_contrast,
                0,
                255,
                cv2.THRESH_BINARY +
                cv2.THRESH_OTSU
            )

            candidates.append(
                thresh2
            )

            # Variant 3
            candidates.append(
                gray
            )

            # ---------------------------------------------
            # OCR
            # ---------------------------------------------

            for candidate in candidates:

                text = reader.readtext(
                    candidate,
                    allowlist=(
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        "0123456789"
                    ),
                    paragraph=False,
                    detail=1
                )

                if not text:
                    continue

                plate_number = clean_ocr_text(
                    text,
                    min_confidence=0.50
                )

                if is_valid_plate(
                    plate_number
                ):

                    # -------------------------------------
                    # BLACKLIST
                    # -------------------------------------

                    if plate_number in BLACKLIST:

                        color = (
                            0,
                            0,
                            255
                        )

                        cv2.putText(
                            output_image,
                            "BLACKLISTED VEHICLE",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color,
                            3
                        )

                    else:

                        color = (
                            0,
                            255,
                            0
                        )

                    # Bounding box
                    cv2.rectangle(
                        output_image,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    # Plate number
                    cv2.putText(
                        output_image,
                        plate_number,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

                    # Confidence
                    cv2.putText(
                        output_image,
                        f"{confidence * 100:.1f}%",
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    detections.append({
                        "plate": plate_number,
                        "confidence": confidence,
                        "blacklisted":
                            plate_number in BLACKLIST
                    })

                    break

    return output_image, detections


# =========================================================
# PDF REPORT
# =========================================================

def create_pdf():

    rows = get_history()

    pdf_path = "NumberPlateReport.pdf"

    c = canvas.Canvas(
        pdf_path,
        pagesize=letter
    )

    c.setFont(
        "Helvetica-Bold",
        16
    )

    c.drawString(
        150,
        770,
        "Number Plate Detection Report"
    )

    c.setFont(
        "Helvetica-Bold",
        12
    )

    c.drawString(
        40,
        740,
        "ID"
    )

    c.drawString(
        90,
        740,
        "Plate Number"
    )

    c.drawString(
        260,
        740,
        "Date & Time"
    )

    y = 720

    c.setFont(
        "Helvetica",
        10
    )

    for row in rows:

        c.drawString(
            40,
            y,
            str(row[0])
        )

        c.drawString(
            90,
            y,
            str(row[1])
        )

        c.drawString(
            260,
            y,
            str(row[3])
        )

        y -= 20

        if y < 40:

            c.showPage()

            y = 770

            c.setFont(
                "Helvetica",
                10
            )

    c.save()

    return pdf_path


# =========================================================
# LOAD MODELS
# =========================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "YOLO model not found: "
        + MODEL_PATH
    )

    st.stop()


with st.spinner(
    "Loading AI models..."
):

    model, reader = load_models()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🚗 ANPR System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Detection",
        "Dashboard",
        "History",
        "Search",
        "Blacklist",
        "Report"
    ]
)


# =========================================================
# DETECTION PAGE
# =========================================================

if page == "Detection":

    st.title(
        "🚗 Automatic Number Plate Recognition"
    )

    st.write(
        "AI-based vehicle number plate "
        "detection and recognition system."
    )

    uploaded_file = st.file_uploader(
        "Upload Vehicle Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        file_bytes = (
            uploaded_file
            .getvalue()
        )

        image_array = cv2.imdecode(
            __import__("numpy").frombuffer(
                file_bytes,
                dtype=__import__("numpy").uint8
            ),
            cv2.IMREAD_COLOR
        )

        st.subheader(
            "Input Image"
        )

        st.image(
            cv2.cvtColor(
                image_array,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

        if st.button(
            "🔍 Detect Number Plate",
            type="primary"
        ):

            with st.spinner(
                "Detecting number plate..."
            ):

                result_image, detections = (
                    detect_plate(
                        image_array,
                        model,
                        reader
                    )
                )

            st.subheader(
                "Detection Result"
            )

            st.image(
                cv2.cvtColor(
                    result_image,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            if detections:

                for detection in detections:

                    plate = detection[
                        "plate"
                    ]

                    confidence = detection[
                        "confidence"
                    ]

                    if detection[
                        "blacklisted"
                    ]:

                        st.error(
                            f"🚨 BLACKLISTED VEHICLE: "
                            f"{plate}"
                        )

                    else:

                        st.success(
                            f"✅ Plate: {plate}"
                        )

                    st.write(
                        f"YOLO Confidence: "
                        f"{confidence * 100:.2f}%"
                    )

                    # Save result
                    save_plate(
                        plate,
                        uploaded_file.name
                    )

            else:

                st.warning(
                    "No valid number plate detected."
                )


# =========================================================
# DASHBOARD
# =========================================================

elif page == "Dashboard":

    st.title(
        "📊 Dashboard"
    )

    rows = get_history()

    total = len(rows)

    unique = len(
        set(
            row[1]
            for row in rows
        )
    )

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    today_count = sum(
        1
        for row in rows
        if str(row[3]).startswith(today)
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Detections",
        total
    )

    col2.metric(
        "Unique Vehicles",
        unique
    )

    col3.metric(
        "Today's Detections",
        today_count
    )

    st.subheader(
        "Recent Detections"
    )

    if rows:

        st.dataframe(
            [
                {
                    "ID": row[0],
                    "Plate Number": row[1],
                    "Image": row[2],
                    "Date & Time": row[3]
                }
                for row in rows[:10]
            ],
            use_container_width=True
        )

    else:

        st.info(
            "No detections available."
        )


# =========================================================
# HISTORY
# =========================================================

elif page == "History":

    st.title(
        "📜 Detection History"
    )

    rows = get_history()

    if rows:

        st.dataframe(
            [
                {
                    "ID": row[0],
                    "Plate Number": row[1],
                    "Image": row[2],
                    "Date & Time": row[3]
                }
                for row in rows
            ],
            use_container_width=True
        )

    else:

        st.info(
            "No detection history."
        )


# =========================================================
# SEARCH
# =========================================================

elif page == "Search":

    st.title(
        "🔎 Search Vehicle"
    )

    search_plate = st.text_input(
        "Enter Number Plate"
    )

    if st.button(
        "Search"
    ):

        rows = get_history()

        results = [
            row
            for row in rows
            if search_plate.upper()
            in row[1].upper()
        ]

        if results:

            st.dataframe(
                [
                    {
                        "ID": row[0],
                        "Plate Number": row[1],
                        "Image": row[2],
                        "Date & Time": row[3]
                    }
                    for row in results
                ],
                use_container_width=True
            )

        else:

            st.warning(
                "No matching vehicle found."
            )


# =========================================================
# BLACKLIST
# =========================================================

elif page == "Blacklist":

    st.title(
        "🚨 Blacklisted Vehicles"
    )

    for plate in BLACKLIST:

        st.error(
            f"🚨 {plate}"
        )


# =========================================================
# REPORT
# =========================================================

elif page == "Report":

    st.title(
        "📄 Number Plate Report"
    )

    rows = get_history()

    st.write(
        f"Total records: {len(rows)}"
    )

    if st.button(
        "Generate PDF Report"
    ):

        pdf_path = create_pdf()

        with open(
            pdf_path,
            "rb"
        ) as file:

            st.download_button(
                label="⬇️ Download PDF Report",
                data=file,
                file_name="NumberPlateReport.pdf",
                mime="application/pdf"
            )