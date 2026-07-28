from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for,
    send_file,
    Response,
    session
)

from ultralytics import YOLO
from werkzeug.utils import secure_filename
import easyocr
import cv2
import numpy as np
import sqlite3
import os
import re
import difflib
import playsound

from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

PLATE_PATTERNS = [
    # Normal Indian plate
    r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$",

    # BH Series: 22BH6517A
    r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"
]


def _digits_to_letters(segment):
    """Fix OCR mistakes where a digit was misread instead of a letter."""
    return (
        segment
        .replace("0", "O")
        .replace("1", "I")
        .replace("2", "Z")
        .replace("5", "S")
        .replace("6", "G")
        .replace("8", "B")
    )


def _letters_to_digits(segment):
    """Fix OCR mistakes where a letter was misread instead of a digit."""
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


VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH",
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB",
    "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB", "AN", "CH",
    "DN", "DD", "DL", "JK", "LA", "LD", "PY"
}


def _correct_state_code(state):
    """Correct a 2-letter state code against the real list of Indian
    RTO codes when it's one letter off (e.g. OCR reading H as I gives
    'IR' instead of 'HR'). Letter-to-letter confusions like this
    aren't digit/letter shape swaps, so _digits_to_letters can't
    catch them - this uses the known valid code list instead. Only
    corrects when there's exactly one close match, to avoid guessing
    wrong when it's ambiguous.
    """
    if state in VALID_STATE_CODES:
        return state

    matches = difflib.get_close_matches(state, VALID_STATE_CODES, n=2, cutoff=0.5)

    if len(matches) == 1:
        return matches[0]

    return state


def fix_indian_plate(plate):
    plate = plate.upper()

    # Remove unwanted characters
    plate = re.sub(r"[^A-Z0-9]", "", plate)

    # Remove IND if OCR reads it
    if plate.startswith("IND"):
        plate = plate[3:]

    # -----------------------------------
    # BH SERIES
    # Example: 22BH6517A
    # -----------------------------------
    if len(plate) >= 2 and "BH" in plate:
        bh_index = plate.find("BH")

        if bh_index == 2:

            prefix = plate[:2]

            # Correct OCR mistakes before BH (should be digits)
            prefix = _letters_to_digits(prefix)

            rest = plate[4:]

            # Convert number section
            number_part = ""
            letter_part = ""

            for char in rest:
                if char.isdigit():
                    number_part += char
                elif char.isalpha():
                    letter_part += char

            # BH format
            if len(number_part) >= 4:
                number_part = number_part[:4]

            if letter_part:
                letter_part = letter_part[:2]

            return prefix + "BH" + number_part + letter_part

    # -----------------------------------
    # NORMAL INDIAN PLATE
    # Layout: LL DD LLL DDDD  (series is 1-3 letters, so total
    # length varies between 9 and 11 characters). Instead of
    # correcting fixed character positions (which breaks for
    # plates with a 1 or 2 letter series), split the plate into
    # its real segments and correct each segment by what it
    # should contain.
    # Example: MH19EQ0009, MH12DE1433, MH12E1433
    # -----------------------------------
    if 9 <= len(plate) <= 11:

        state = _correct_state_code(_digits_to_letters(plate[0:2]))
        rto = _letters_to_digits(plate[2:4])
        series = _digits_to_letters(plate[4:-4])
        number = _letters_to_digits(plate[-4:])

        return state + rto + series + number

    # Fallback: at least fix the state code (first two letters)
    if len(plate) >= 2:
        plate = _digits_to_letters(plate[:2]) + plate[2:]

    return plate

app = Flask(__name__)
app.secret_key = "number_plate_secret_key"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )
# Load AI Models
model = YOLO("models/best.pt")

# Sanity check: this app only works if models/best.pt is a model
# that was actually trained to detect NUMBER PLATES. If it's a
# generic pretrained checkpoint (e.g. stock yolov8n.pt), it detects
# COCO objects like "car"/"person" instead, and every OCR read below
# will run on the wrong region of the image - producing garbage text
# that occasionally happens to match the plate regex by chance. That
# looks exactly like "wrong number detected" without ever raising an
# error. Loading yolov8n.pt in place of a trained best.pt is the most
# likely explanation if plates are misread on both upload and live
# camera.
_plate_like_classes = {
    name for name in model.names.values()
    if "plate" in name.lower()
}
if not _plate_like_classes:
    print(
        "WARNING: models/best.pt has no class with 'plate' in its "
        "name (classes found: {}). This looks like a generic "
        "pretrained YOLO checkpoint, not a number-plate detector. "
        "Detections will be run on the wrong region of the image "
        "and OCR output will be unreliable.".format(list(model.names.values()))
    )

reader = easyocr.Reader(['en'], gpu=False)


blacklist = [
    "MH31AB1234",
    "HR98AA7777"
]


def is_logged_in():
    """Return True when the admin has an active login session."""
    return "user" in session


def get_db_connection():
    """Create a SQLite connection that allows row.field access in templates."""
    conn = sqlite3.connect("plates.db")
    conn.row_factory = sqlite3.Row
    return conn


def clean_plate_text(text_items, min_confidence=0.60):
    """Convert EasyOCR results into a corrected Indian plate number."""
    plate_number = ""

    # EasyOCR does not guarantee reading order when a plate is split into
    # multiple text boxes. Sort by top (y) then left (x) coordinate of each
    # box so fragments are joined in the correct left-to-right, top-to-bottom
    # order instead of whatever order EasyOCR happened to return them in.
    filtered_items = [item for item in text_items if item[2] >= min_confidence]

    def box_position(item):
        box = item[0]  # list of 4 (x, y) corner points
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        return (min(ys), min(xs))

    filtered_items.sort(key=box_position)

    for item in filtered_items:
        # EasyOCR item format: (box_coordinates, detected_text, confidence)
        plate_number += item[1]

    plate_number = re.sub(r"[^A-Z0-9]", "", plate_number.upper())
    plate_number = plate_number.replace("IND", "").strip()

    return fix_indian_plate(plate_number)


def is_valid_plate(plate_number):
    """Validate normal Indian and BH-series number plates.

    Shape alone (2 letters + 2 digits + series + 4 digits) isn't
    enough: OCR confusions like M<->H produce strings such as
    'HH12DE1433' that match the shape perfectly but start with a
    state code ('HH') that doesn't exist. Previously these were
    accepted as "valid" and saved straight to the database. Now the
    state-code segment of a normal-format plate is also checked
    against the real list of Indian RTO codes, so a shape-correct
    but nonexistent state code is rejected instead of silently saved -
    forcing the caller to try another OCR candidate/frame instead.
    """

    plate_number = plate_number.upper().strip()

    for pattern in PLATE_PATTERNS:
        if re.match(pattern, plate_number):
            if "BH" in pattern:
                return True
            if plate_number[0:2] in VALID_STATE_CODES:
                return True

    return False


def find_similar_plate(cur, plate_number, threshold=0.85):
    """Return an existing plate_number that's a near-duplicate of this
    one (e.g. OCR read S instead of 5, or H instead of I on a repeat
    detection of the same vehicle), or None if there's no close match.

    A plain exact-match check misses this because two OCR passes on
    the same physical plate can legitimately produce slightly
    different strings.
    """
    cur.execute("SELECT DISTINCT plate_number FROM plates")
    existing_plates = [row[0] for row in cur.fetchall()]

    for existing in existing_plates:
        if existing == plate_number:
            return existing

        similarity = difflib.SequenceMatcher(
            None, existing, plate_number
        ).ratio()

        if similarity >= threshold:
            return existing

    return None


def make_safe_upload_name(filename):
    """Prevent unsafe filenames and avoid overwriting older uploads."""
    safe_name = secure_filename(filename)
    name, ext = os.path.splitext(safe_name)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

    return f"{name}_{timestamp}{ext.lower()}"


# -------------Login Route----------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "admin123":

            session["user"] = username
            return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            error="Invalid Username or Password"
        )

    return render_template("login.html")

# ---------Logout Route-------------------
@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))
# ---------------- HOME ----------------
@app.route("/")
def home():

    if not is_logged_in():
        return redirect(url_for("login"))

    conn = sqlite3.connect("plates.db")
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM plates")
    total = cur.fetchone()[0]

    conn.close()

    return render_template(
        "index.html",
        total=total
    )

# ---------------- HISTORY ----------------

@app.route("/history")
def history():

    if not is_logged_in():
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM plates
        ORDER BY id DESC
    """)

    records = cur.fetchall()

    conn.close()

    return render_template(
        "history.html",
        records=records
    )

@app.route("/view/<int:id>")
def view(id):

    if not is_logged_in():
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM plates
        WHERE id=?
        """,
        (id,)
    )

    row = cur.fetchone()

    conn.close()

    if row is None:
        return "Record Not Found"

    return render_template(
        "view.html",
        row=row,
        blacklist=blacklist
    )

# ---------------- DELETE ----------------

@app.route("/delete/<int:id>", methods=["POST"])
def delete(id):

    if not is_logged_in():
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT image_name FROM plates WHERE id=?",
        (id,)
    )

    row = cur.fetchone()

    if row:

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            row["image_name"]
        )

        cur.execute(
            "DELETE FROM plates WHERE id=?",
            (id,)
        )

        conn.commit()

        if os.path.exists(image_path):
            os.remove(image_path)

    conn.close()

    return redirect(url_for("history"))

# ---------------- EDIT ----------------

@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit(id):

    if not is_logged_in():
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()

    if request.method == "POST":

        plate = request.form["plate"].strip().upper()
        # Normalize to match how every other entry point stores plates
        # (alphanumeric only, no stray spaces/symbols).
        plate = re.sub(r"[^A-Z0-9]", "", plate)

        cur.execute(
            """
            UPDATE plates
            SET plate_number=?
            WHERE id=?
            """,
            (plate, id)
        )

        conn.commit()
        conn.close()

        return redirect(url_for("history"))

    cur.execute(
        """
        SELECT *
        FROM plates
        WHERE id=?
        """,
        (id,)
    )

    row = cur.fetchone()

    conn.close()

    if row is None:
        return "Record Not Found"

    return render_template(
        "edit.html",
        row=row
    )
# ---------------- DASHBOARD ----------------

# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():

    if not is_logged_in():
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()

    # Total detections
    cur.execute("SELECT COUNT(*) FROM plates")
    total = cur.fetchone()[0]

    # Unique vehicles
    cur.execute("SELECT COUNT(DISTINCT plate_number) FROM plates")
    unique = cur.fetchone()[0]

    # Today detections
    cur.execute("""
        SELECT COUNT(*)
        FROM plates
        WHERE DATE(date_time)=DATE("now","localtime")
    """)
    today = cur.fetchone()[0]

    # Recent detections
    cur.execute("""
        SELECT *
        FROM plates
        ORDER BY id DESC
        LIMIT 5
    """)
    recent = cur.fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        total=total,
        unique=unique,
        today=today,
        recent=recent
    )
# ---------------- ANALYTICS ----------------

@app.route("/analytics")
def analytics():

    if not is_logged_in():
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM plates")
    total = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(DISTINCT plate_number) FROM plates"
    )
    unique = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*)
        FROM plates
        WHERE DATE(date_time)=DATE('now','localtime')
    """)

    today = cur.fetchone()[0]

    cur.execute("""
        SELECT
            plate_number,
            COUNT(*) AS total
        FROM plates
        GROUP BY plate_number
        ORDER BY total DESC
        LIMIT 5
    """)

    top = cur.fetchall()

    labels = []
    values = []

    for row in top:
        labels.append(row["plate_number"])
        values.append(row["total"])

    conn.close()

    return render_template(
        "analytics.html",
        total=total,
        unique=unique,
        today=today,
        top=top,
        labels=labels,
        values=values
    )

# ---------------- SEARCH ----------------

@app.route("/search", methods=["GET", "POST"])
def search():
    if not is_logged_in():
        return redirect(url_for("login"))
    records = []

    if request.method == "POST":

        plate = request.form["plate"].strip().upper()

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT *
            FROM plates
            WHERE plate_number LIKE ?
            ORDER BY id DESC
        """, ('%' + plate + '%',))

        records = cur.fetchall()

        conn.close()

    return render_template(
        "search.html",
        records=records
    )

# ---------------- IMAGE ----------------

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ---------------- PDF REPORT ----------------

@app.route("/report")
def report():

    if not is_logged_in():
        return redirect(url_for("login"))

    conn = sqlite3.connect("plates.db")
    cur = conn.cursor()

    cur.execute("""
        SELECT id, plate_number, date_time
        FROM plates
        ORDER BY id DESC
    """)

    rows = cur.fetchall()

    conn.close()

    pdf_file = "NumberPlateReport.pdf"

    c = canvas.Canvas(pdf_file, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(150, 770, "Number Plate Detection Report")

    c.setFont("Helvetica-Bold", 12)

    c.drawString(40, 740, "ID")
    c.drawString(90, 740, "Plate Number")
    c.drawString(260, 740, "Date & Time")

    y = 720

    c.setFont("Helvetica", 11)

    for row in rows:

        c.drawString(40, y, str(row[0]))
        c.drawString(90, y, row[1])
        c.drawString(260, y, str(row[2]))

        y -= 20

        if y < 40:

            c.showPage()

            y = 770

    c.save()

    return send_file(
        pdf_file,
        as_attachment=True
    )

# ---------------- DETECTION ----------------

@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    if not allowed_file(file.filename):
        return "Only JPG, JPEG and PNG files are allowed."

    filename = make_safe_upload_name(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)

    # Detect plate
    results = model.predict(
    filepath,
    conf=0.30,
    imgsz=640,
    verbose=False
)

    image = cv2.imread(filepath)

    if image is None:
        return "Invalid image file"

    plate_number = "NOT DETECTED"
    detected_plate = False

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            # Pad the crop a little so tight YOLO boxes don't clip
            # the edge characters of the plate.
            pad_x = int((x2 - x1) * 0.06)
            pad_y = int((y2 - y1) * 0.15)
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(image.shape[1], x2 + pad_x)
            py2 = min(image.shape[0], y2 + pad_y)

            plate = image[py1:py2, px1:px2]
            if plate.size == 0:
                continue

            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

            # Upscale
            gray = cv2.resize(
                gray,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            # -------------------------
            # OCR - try a couple of preprocessing variants and
            # keep the first one that produces a valid plate.
            # equalizeHist + bilateralFilter over-processed clean
            # plates into unreadable noise, so we avoid heavy
            # global-contrast operations here.
            # -------------------------

            candidates = []

            # Variant 1: light blur + OTSU threshold
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh1 = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            candidates.append(thresh1)

            # Variant 2: CLAHE (local contrast) + OTSU threshold,
            # helps with uneven lighting/glare on the plate
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            local_contrast = clahe.apply(gray)
            _, thresh2 = cv2.threshold(
                local_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            candidates.append(thresh2)

            # Variant 3: plain upscaled grayscale, no threshold at all
            candidates.append(gray)

            plate_number = "NOT DETECTED"
            found_valid = False

            for candidate_img in candidates:

                text = reader.readtext(
                    candidate_img,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    paragraph=False,
                    detail=1
                )

                print("RAW OCR:", text)

                if len(text) == 0:
                    continue

                candidate_plate = clean_plate_text(
                    text,
                    min_confidence=0.40
                )

                print("OCR:", candidate_plate)

                if is_valid_plate(candidate_plate):
                    plate_number = candidate_plate
                    found_valid = True
                    break
                else:
                    # Keep the last attempt around so the user at
                    # least sees what OCR read, even if invalid.
                    plate_number = candidate_plate

            if not found_valid:
                print("Invalid Plate:", plate_number)
                continue

            detected_plate = True

            # -------------------------
            # SUCCESS
            # -------------------------

            detected_plate = True

            print("FINAL DETECTED PLATE:", plate_number)

            if plate_number in blacklist:
                color = (0, 0, 255)

                cv2.putText(
                    image,
                    "BLACKLISTED VEHICLE",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3
                )
            else:
                color = (0, 255, 0)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            confidence_text = f"{confidence * 100:.1f}%"

            cv2.putText(
                image,
                plate_number,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                image,
                confidence_text,
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
            break

        if detected_plate:
            break
                    

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Save processed image (rectangle + plate text + confidence)
    cv2.imwrite(filepath, image)


    # ---------------- DATABASE ----------------

    if not detected_plate:
        return render_template(
            "result.html",
            plate=plate_number,
            image=filename,
            message="No readable number plate detected"
        )

    conn = sqlite3.connect("plates.db")
    cur = conn.cursor()

    cur.execute(
        "SELECT id FROM plates WHERE plate_number=?",
        (plate_number,)
    )

    existing = cur.fetchone()
    similar_plate = None if existing else find_similar_plate(cur, plate_number)

    if existing is None and similar_plate is None:

        cur.execute(
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
                filename,
                current_time
            )
        )

        conn.commit()

        message = "New plate saved successfully"

    else:

        message = "Plate already exists"
        if similar_plate:
            message = f"Plate already exists (matched similar plate: {similar_plate})"

    conn.close()

    return render_template(
        "result.html",
        plate=plate_number,
        image=filename,
        message=message
    )
# ---------------- LIVE CAMERA ----------------

@app.route("/live")
def live():
    if not is_logged_in():
        return redirect(url_for("login"))

    return render_template("live.html")


# ---------------- VIDEO STREAM ----------------

camera = None

# Tracks the last time each plate number was saved, so the same
# vehicle sitting in frame for several seconds doesn't get re-saved
# on every frame. This MUST be defined here (module scope) - the
# `global last_detected` inside generate_frames() only tells Python
# to use this variable, it does not create it. Without this line the
# app crashed with NameError the first time any plate was recognized
# on the live feed, killing the video stream.
last_detected = {}


def get_camera():
    """Lazily (re)open the webcam.

    The camera used to be opened once at import time with the
    Windows-only cv2.CAP_DSHOW backend. On Linux/macOS/containers (or
    any machine with no webcam attached), that open silently fails,
    and since it only happened once, the stream was dead forever with
    no retry and no visible error. This opens on first use and will
    retry if the connection was lost or never succeeded.
    """
    global camera

    if camera is not None and camera.isOpened():
        return camera

    # CAP_DSHOW only makes sense on Windows; let OpenCV pick the
    # right backend automatically everywhere else.
    if os.name == "nt":
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(0)

    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera Opened:", camera.isOpened())
    return camera


def _camera_unavailable_frame():
    """A placeholder JPEG frame shown when no webcam is available,
    so the browser gets a clear message instead of a dead stream."""
    blank = np.zeros((480, 640, 3), dtype="uint8")
    cv2.putText(
        blank,
        "Camera not available",
        (60, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )
    _, buffer = cv2.imencode(".jpg", blank)
    return buffer.tobytes()


def generate_frames():

    global last_detected

    cam = get_camera()

    if not cam.isOpened():
        # No webcam available (common on servers/containers with no
        # camera hardware, or wrong OS backend). Show one clear
        # placeholder frame instead of an endless dead stream.
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            _camera_unavailable_frame() +
            b'\r\n'
        )
        return

    while True:
        success, frame = cam.read()

        if not success:
            # Lost connection mid-stream - try to recover once
            # instead of silently ending the generator forever.
            cam.release()
            cam = get_camera()
            if not cam.isOpened():
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    _camera_unavailable_frame() +
                    b'\r\n'
                )
                return
            continue
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Detect plate candidates in the current camera frame.
        # conf lowered from 0.5: live frames (motion blur, distance,
        # lighting) naturally score lower confidence than a still
        # upload, so 0.5+0.70 (see below) was filtering out every
        # single detection before OCR ever ran.
        results = model.predict(
            frame,
            imgsz=640,
            conf=0.35,
            verbose=False
        )
        for result in results:

            for box in result.boxes:
                print("Plate Detected")

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Was 0.70 - this alone was silently dropping every
                # detection before "Running OCR..." could ever print.
                # Kept as a light sanity filter, not a hard gate.
                if confidence < 0.40:
                    continue

                print("Running OCR...")

                # Pad the crop slightly so a tight YOLO box doesn't
                # clip the outer characters of the plate - the upload
                # route already did this, but the live feed didn't,
                # which was cutting off the leading state code letters
                # or the trailing digit(s) on many frames.
                pad_x = int((x2 - x1) * 0.10)
                pad_y = int((y2 - y1) * 0.20)
                px1 = max(0, x1 - pad_x)
                py1 = max(0, y1 - pad_y)
                px2 = min(frame.shape[1], x2 + pad_x)
                py2 = min(frame.shape[0], y2 + pad_y)

                plate = frame[py1:py2, px1:px2]

                if plate.size == 0:
                    continue
                
                gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(
                    gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
                )

                # Same multi-candidate strategy as the upload route:
                # try a couple of preprocessing variants and keep the
                # first one that yields a plate matching the expected
                # format. A single fixed threshold (the old behavior
                # here) works far worse on live frames, which have
                # more variable lighting/glare than a still upload.
                candidates = []

                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh1 = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                candidates.append(thresh1)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                local_contrast = clahe.apply(gray)
                _, thresh2 = cv2.threshold(
                    local_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                candidates.append(thresh2)

                candidates.append(gray)

                plate_number = "NOT DETECTED"
                found_valid = False

                for candidate_img in candidates:
                    text = reader.readtext(
                        candidate_img,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        paragraph=False,
                        detail=1
                    )

                    print("OCR Result:", text)

                    if not text:
                        continue

                    candidate_plate = clean_plate_text(text, min_confidence=0.40)

                    if is_valid_plate(candidate_plate):
                        plate_number = candidate_plate
                        found_valid = True
                        break
                    else:
                        plate_number = candidate_plate

                if not found_valid:
                    print("Invalid Plate:", plate_number)
                    continue

                print("Final Plate:", plate_number)

                current = datetime.now()

                if plate_number in last_detected:
                    diff = (current - last_detected[plate_number]).total_seconds()
                    if diff < 10:
                        continue

                last_detected[plate_number] = current

                

                if plate_number:
                    confidence_text = f"{confidence * 100:.1f}%"
                    
                    # Decide rectangle color
                    if plate_number in blacklist:

                        alarm_path = os.path.join("static", "css", "fahhhhh.mp3")
                        try:
                            playsound.playsound(alarm_path, block=False)
                        except Exception as e:
                            # playsound is notoriously platform-fragile
                            # (missing audio backend, missing file, the
                            # `block` kwarg not being supported on this
                            # platform's backend, etc). Never let an
                            # alarm sound failure take down the live
                            # video feed.
                            print("Alarm sound failed:", e)

                        

                        color = (0, 0, 255)

                        cv2.putText(
                            frame,
                            "BLACKLISTED VEHICLE",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color,
                            3
                        )

                    else:
                        
                        color = (0, 255, 0)

                    # Draw rectangle
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    label = f"{plate_number} ({confidence * 100:.1f}%)"

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                conn = sqlite3.connect("plates.db")
                cur = conn.cursor()

                cur.execute(
                    "SELECT id FROM plates WHERE plate_number=?",
                    (plate_number,)
                )

                exists = cur.fetchone()
                similar_plate = None if exists else find_similar_plate(cur, plate_number)

                if exists is None and similar_plate is None:

                    filename = f"{plate_number}.jpg"

                    cv2.imwrite(
                        os.path.join(app.config["UPLOAD_FOLDER"], filename),
                        frame
                    )

                    cur.execute(
                        """
                        INSERT INTO plates
                        (plate_number, image_name, date_time)
                        VALUES (?, ?, ?)
                        """,
                        (
                            plate_number,
                            filename,
                            current_time
                        )
                    )

                    conn.commit()
                    print("Saved:", plate_number)
                elif similar_plate:
                    print(f"Skipped near-duplicate of {similar_plate}:", plate_number)

                conn.close()



        # Convert to JPEG
        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )
# ---------------- VIDEO FEED ----------------

@app.route("/video_feed")
def video_feed():
    if not is_logged_in():
        return redirect(url_for("login"))

    print("video_feed called")

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=False, threaded=True)