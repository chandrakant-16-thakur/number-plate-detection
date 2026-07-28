from ultralytics import YOLO
import cv2
import easyocr
import re


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


def fix_indian_plate(plate):
    """Correct common OCR letter/digit confusions.

    Rather than correcting fixed character positions (which only
    works for one exact plate length), this splits the plate into
    its real segments - state (2 letters), RTO code (2 digits),
    series (1-3 letters), number (4 digits) - and corrects each
    segment according to what it should contain. This handles all
    valid series lengths, e.g. MH12DE1433 (2-letter series) and
    MH12E1433 (1-letter series).
    """
    plate = plate.upper()
    plate = re.sub(r"[^A-Z0-9]", "", plate)

    if plate.startswith("IND"):
        plate = plate[3:]

    # BH series, e.g. 22BH6517A
    if len(plate) >= 2 and "BH" in plate:
        bh_index = plate.find("BH")

        if bh_index == 2:
            prefix = _letters_to_digits(plate[:2])
            rest = plate[4:]

            number_part = "".join(c for c in rest if c.isdigit())[:4]
            letter_part = "".join(c for c in rest if c.isalpha())[:2]

            return prefix + "BH" + number_part + letter_part

    # Normal plate: LL DD LLL DDDD (series is 1-3 letters, so total
    # length is 9-11 characters)
    if 9 <= len(plate) <= 11:
        state = _digits_to_letters(plate[0:2])
        rto = _letters_to_digits(plate[2:4])
        series = _digits_to_letters(plate[4:-4])
        number = _letters_to_digits(plate[-4:])
        return state + rto + series + number

    # Fallback: at least fix the state code
    if len(plate) >= 2:
        plate = _digits_to_letters(plate[:2]) + plate[2:]

    return plate


PLATE_PATTERNS = [
    r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$",
    r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$",
]


def is_valid_plate(plate_number):
    plate_number = plate_number.upper().strip()
    return any(re.match(pattern, plate_number) for pattern in PLATE_PATTERNS)


def clean_ocr_text(text_items, min_confidence=0.50):
    """Join EasyOCR fragments in left-to-right, top-to-bottom order."""
    filtered = [item for item in text_items if item[2] >= min_confidence]

    def box_position(item):
        box = item[0]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (min(ys), min(xs))

    filtered.sort(key=box_position)

    plate_number = "".join(item[1] for item in filtered)
    plate_number = re.sub(r"[^A-Z0-9]", "", plate_number.upper())
    return fix_indian_plate(plate_number)


MODEL_PATH = "models/best.pt"
IMAGE_PATH = "uploads/car.jpg"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Sanity check: make sure this is actually a plate-detector model,
# not a generic pretrained checkpoint (e.g. stock yolov8n.pt) that
# only knows COCO classes like "car"/"person". If it's generic, YOLO
# will localize the wrong region and every OCR read below will be
# garbage that occasionally slips past the plate regex by chance.
_plate_like_classes = {
    name for name in model.names.values() if "plate" in name.lower()
}
if not _plate_like_classes:
    print(
        "WARNING:", MODEL_PATH, "has no class with 'plate' in its name "
        f"(classes found: {list(model.names.values())}). This looks like "
        "a generic pretrained YOLO checkpoint, not a number-plate detector."
    )

# Load EasyOCR
reader = easyocr.Reader(['en'], gpu=False)


def detect_plate(image_path):
    results = model.predict(image_path, conf=0.40, save=True)
    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image:", image_path)
        return None

    for result in results:
        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Pad the crop slightly so a tight YOLO box doesn't clip
            # the outer characters of the plate.
            pad_x = int((x2 - x1) * 0.06)
            pad_y = int((y2 - y1) * 0.15)
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(image.shape[1], x2 + pad_x)
            py2 = min(image.shape[0], y2 + pad_y)

            plate = image[py1:py2, px1:px2]
            if plate.size == 0:
                continue

            cv2.imwrite("plate.jpg", plate)

            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            # Try a few preprocessing variants and use the first one
            # that yields a plate matching the expected format.
            # NOTE: equalizeHist + bilateralFilter tends to blow out
            # contrast on already high-contrast plates and destroy
            # the characters - avoid heavy global-contrast operations.
            candidates = []

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            candidates.append(("blur+otsu", thresh1))

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            local_contrast = clahe.apply(gray)
            _, thresh2 = cv2.threshold(local_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            candidates.append(("clahe+otsu", thresh2))

            candidates.append(("raw_gray", gray))

            best_plate = "NOT DETECTED"

            for label, candidate_img in candidates:
                text = reader.readtext(
                    candidate_img,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    paragraph=False,
                    detail=1
                )

                if not text:
                    continue

                plate_number = clean_ocr_text(text, min_confidence=0.50)
                print(f"[{label}] OCR raw={text} -> cleaned={plate_number}")

                best_plate = plate_number

                if is_valid_plate(plate_number):
                    print("\n==============================")
                    print(f"Valid Plate ({label}):", plate_number)
                    print("==============================")
                    return plate_number

            print("\n==============================")
            print("No valid plate found. Last OCR attempt:", best_plate)
            print("==============================")

    return None


if __name__ == "__main__":
    detect_plate(IMAGE_PATH)