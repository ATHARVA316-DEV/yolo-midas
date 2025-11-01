import cv2
import numpy as np
import pytesseract
import pyttsx3
import time

# =======================
# CONFIGURATION
# =======================
LOWER_YELLOW = np.array([18, 100, 100])
UPPER_YELLOW = np.array([35, 255, 255])
MIN_AREA = 1500
CENTER_TOLERANCE = 60
CIRCULARITY_THRESHOLD = 0.6

# Tesseract path (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Initialize speech
engine = pyttsx3.init()
engine.setProperty('rate', 170)

def speak(text):
    print(f"[SPEAK] {text}")
    engine.say(text)
    engine.runAndWait()

# =======================
# DETECTION FUNCTIONS
# =======================
def detect_yellow_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        return None
    return largest

def classify_pattern(roi_mask):
    """Detect dots (warning) vs bars (directional)."""
    gray = roi_mask.copy()
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=10)
    
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circular_count = 0
    for c in contours:
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        if per == 0: continue
        circularity = 4 * np.pi * (area / (per * per))
        if 0.5 < circularity < 1.2 and 50 < area < 2000:
            circular_count += 1

    if circular_count >= 3 and (lines is None or len(lines) < 6):
        return "warning (dots)"
    elif lines is not None and len(lines) >= 6:
        return "directional (bars)"
    else:
        return "unknown"

def read_sign_text(frame):
    """Extract readable text using OCR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h)
        if 0.5 < aspect < 3 and w > 100 and h > 40:
            roi = frame[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi, lang='eng', config='--psm 6').strip()
            if len(text) > 2 and any(ch.isalnum() for ch in text):
                return text
    return None

# =======================
# MAIN LOOP
# =======================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not detected!")
    exit()

last_pattern = ""
last_sign = ""
last_speak_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]

    # --- Line Detection ---
    mask = detect_yellow_mask(frame)
    contour = get_largest_contour(mask)
    guidance = "No line"
    pattern_label = "unknown"

    if contour is not None:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
        else:
            cx = w // 2

        dx = cx - (w // 2)
        if abs(dx) < CENTER_TOLERANCE:
            guidance = "Go straight"
        elif dx < 0:
            guidance = "Move left"
        else:
            guidance = "Move right"

        x, y, ww, hh = cv2.boundingRect(contour)
        roi_mask = mask[y:y+hh, x:x+ww]
        pattern_label = classify_pattern(roi_mask)

        cv2.drawContours(frame, [contour], -1, (0,255,0), 2)
        cv2.circle(frame, (cx, h//2), 6, (0,0,255), -1)
        cv2.putText(frame, guidance, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Pattern: {pattern_label}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Speak guidance occasionally
        now = time.time()
        if now - last_speak_time > 1.5:
            speak(f"{guidance}, {pattern_label}")
            last_speak_time = now

    # --- OCR Sign Reading ---
    sign_text = read_sign_text(frame)
    if sign_text and sign_text != last_sign:
        last_sign = sign_text
        cv2.putText(frame, f"Sign: {sign_text}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        speak(f"Sign detected: {sign_text}")

    cv2.imshow("Smart Line & Sign Assistant", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
