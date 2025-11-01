import cv2
import numpy as np
import pytesseract
import pyttsx3
import time
import threading
from collections import deque
from typing import Optional, Tuple
import platform

# =======================
# CONFIGURATION
# =======================
class Config:
    """Centralized configuration management"""
    def __init__(self):
        # Color detection
        self.LOWER_YELLOW = np.array([18, 100, 100])
        self.UPPER_YELLOW = np.array([35, 255, 255])
        
        # Line detection
        self.MIN_AREA = 1500
        self.CENTER_TOLERANCE = 60
        self.CIRCULARITY_THRESHOLD = 0.6
        
        # Camera
        self.CAMERA_INDEX = 0
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
        
        # Speech
        self.SPEECH_RATE = 170
        self.SPEAK_INTERVAL = 2.0
        
        # OCR
        self.OCR_MIN_TEXT_LENGTH = 2
        self.OCR_CONFIDENCE_THRESHOLD = 60
        
        # Performance
        self.HISTORY_SIZE = 5
        self.SMOOTHING_ENABLED = True

config = Config()

# Platform-specific Tesseract setup
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =======================
# TEXT-TO-SPEECH HANDLER
# =======================
class SpeechHandler:
    """Thread-safe speech synthesis"""
    def __init__(self, rate: int = 170):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.speech_queue = deque(maxlen=3)
        self.is_speaking = False
        self.lock = threading.Lock()
        
    def speak_async(self, text: str):
        """Non-blocking speech synthesis"""
        if not text:
            return
            
        with self.lock:
            # Avoid duplicate announcements
            if self.speech_queue and self.speech_queue[-1] == text:
                return
            self.speech_queue.append(text)
        
        if not self.is_speaking:
            threading.Thread(target=self._process_queue, daemon=True).start()
    
    def _process_queue(self):
        """Process speech queue in background"""
        while self.speech_queue:
            with self.lock:
                if not self.speech_queue:
                    break
                text = self.speech_queue.popleft()
                self.is_speaking = True
            
            print(f"[SPEAK] {text}")
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"[ERROR] Speech synthesis failed: {e}")
            
            time.sleep(0.3)  # Brief pause between announcements
        
        with self.lock:
            self.is_speaking = False

# =======================
# DETECTION CLASSES
# =======================
class LineDetector:
    """Handles yellow line detection and guidance"""
    def __init__(self, cfg: Config):
        self.config = cfg
        self.center_history = deque(maxlen=cfg.HISTORY_SIZE)
        
    def detect_yellow_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create binary mask for yellow regions"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.LOWER_YELLOW, self.config.UPPER_YELLOW)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def get_largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Find the largest valid contour"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.config.MIN_AREA:
            return None
        
        return largest
    
    def get_smoothed_center(self, cx: int) -> int:
        """Apply moving average smoothing to center position"""
        if not self.config.SMOOTHING_ENABLED:
            return cx
        
        self.center_history.append(cx)
        return int(np.mean(self.center_history))
    
    def classify_pattern(self, roi_mask: np.ndarray) -> str:
        """Classify line pattern: dots (warning) vs bars (directional)"""
        if roi_mask.size == 0:
            return "unknown"
        
        # Edge detection for line analysis
        edges = cv2.Canny(roi_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=10)
        
        # Detect circular patterns (dots)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circular_count = 0
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50 or area > 2000:
                continue
            
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if 0.5 < circularity < 1.2:
                circular_count += 1
        
        # Classification logic
        if circular_count >= 3 and (lines is None or len(lines) < 6):
            return "warning dots"
        elif lines is not None and len(lines) >= 6:
            return "directional bars"
        else:
            return "solid line"
    
    def get_guidance(self, cx: int, frame_width: int) -> str:
        """Generate navigation guidance based on line position"""
        cx_smoothed = self.get_smoothed_center(cx)
        frame_center = frame_width // 2
        dx = cx_smoothed - frame_center
        
        if abs(dx) < self.config.CENTER_TOLERANCE:
            return "Go straight"
        elif dx < -self.config.CENTER_TOLERANCE:
            return "Move right"
        else:
            return "Move left"

class SignDetector:
    """Handles text sign detection using OCR"""
    def __init__(self, cfg: Config):
        self.config = cfg
        self.last_sign = ""
        self.last_detection_time = 0
        
    def preprocess_for_ocr(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for better OCR accuracy"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding for varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return denoised
    
    def find_text_regions(self, frame: np.ndarray) -> list:
        """Detect potential text-containing regions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        regions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            
            # Filter for sign-like rectangles
            if 0.5 < aspect < 4 and w > 100 and h > 40:
                regions.append((x, y, w, h))
        
        return regions
    
    def read_sign_text(self, frame: np.ndarray) -> Optional[str]:
        """Extract and validate text from frame"""
        current_time = time.time()
        
        # Limit OCR frequency for performance
        if current_time - self.last_detection_time < 0.5:
            return None
        
        regions = self.find_text_regions(frame)
        
        for x, y, w, h in regions:
            roi = frame[y:y+h, x:x+w]
            preprocessed = self.preprocess_for_ocr(roi)
            
            try:
                # Get detailed OCR data
                data = pytesseract.image_to_data(
                    preprocessed, 
                    lang='eng', 
                    config='--psm 6',
                    output_type=pytesseract.Output.DICT
                )
                
                # Filter by confidence
                text_parts = []
                for i, conf in enumerate(data['conf']):
                    if int(conf) > self.config.OCR_CONFIDENCE_THRESHOLD:
                        text = data['text'][i].strip()
                        if text and any(ch.isalnum() for ch in text):
                            text_parts.append(text)
                
                if text_parts:
                    full_text = ' '.join(text_parts)
                    if len(full_text) >= self.config.OCR_MIN_TEXT_LENGTH:
                        self.last_detection_time = current_time
                        return full_text
                        
            except Exception as e:
                print(f"[ERROR] OCR failed: {e}")
                continue
        
        return None

# =======================
# MAIN APPLICATION
# =======================
class SmartVisionAssistant:
    """Main application controller"""
    def __init__(self):
        self.config = Config()
        self.speech = SpeechHandler(self.config.SPEECH_RATE)
        self.line_detector = LineDetector(self.config)
        self.sign_detector = SignDetector(self.config)
        
        self.last_guidance = ""
        self.last_pattern = ""
        self.last_speak_time = 0
        
        self.cap = None
        self.running = False
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        
        if not self.cap.isOpened():
            print("[ERROR] Camera not detected!")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        
        print("[INFO] Camera initialized successfully")
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame for line and sign detection"""
        h, w = frame.shape[:2]
        
        # --- Line Detection ---
        mask = self.line_detector.detect_yellow_mask(frame)
        contour = self.line_detector.get_largest_contour(mask)
        
        if contour is not None:
            # Calculate center and guidance
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else w // 2
            
            guidance = self.line_detector.get_guidance(cx, w)
            
            # Extract ROI and classify pattern
            x, y, ww, hh = cv2.boundingRect(contour)
            roi_mask = mask[y:y+hh, x:x+ww]
            pattern = self.line_detector.classify_pattern(roi_mask)
            
            # Visualization
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, h//2), 6, (0, 0, 255), -1)
            cv2.line(frame, (w//2, 0), (w//2, h), (255, 0, 0), 1)
            
            # Display info
            cv2.putText(frame, guidance, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Pattern: {pattern}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Voice guidance
            current_time = time.time()
            if (guidance != self.last_guidance or pattern != self.last_pattern) and \
               (current_time - self.last_speak_time > self.config.SPEAK_INTERVAL):
                self.speech.speak_async(f"{guidance}, {pattern}")
                self.last_guidance = guidance
                self.last_pattern = pattern
                self.last_speak_time = current_time
        else:
            cv2.putText(frame, "No line detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # --- Sign Detection ---
        sign_text = self.sign_detector.read_sign_text(frame)
        if sign_text and sign_text != self.sign_detector.last_sign:
            self.sign_detector.last_sign = sign_text
            cv2.putText(frame, f"Sign: {sign_text}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            self.speech.speak_async(f"Sign detected: {sign_text}")
        
        # Display FPS
        fps_text = f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}"
        cv2.putText(frame, fps_text, (w - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main application loop"""
        if not self.initialize_camera():
            return
        
        self.running = True
        print("[INFO] Starting Smart Vision Assistant...")
        print("[INFO] Press 'q' or ESC to quit")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("[WARNING] Failed to read frame")
                    break
                
                # Resize for consistent processing
                frame = cv2.resize(frame, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT))
                
                # Process and display
                processed_frame = self.process_frame(frame)
                cv2.imshow("Smart Line & Sign Assistant", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("[INFO] Shutting down...")
                    break
                elif key == ord('c'):
                    # Calibration mode (adjust color thresholds)
                    print(f"[INFO] Current HSV range: {self.config.LOWER_YELLOW} - {self.config.UPPER_YELLOW}")
                    
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released")

# =======================
# ENTRY POINT
# =======================
if __name__ == "__main__":
    assistant = SmartVisionAssistant()
    assistant.run()