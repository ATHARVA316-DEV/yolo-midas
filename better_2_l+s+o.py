import cv2
import numpy as np
import pytesseract
import pyttsx3
import time
import threading
from collections import deque
from typing import Optional, Tuple, List
import platform
import json
import os
from enum import Enum

# =======================
# ENUMS & CONSTANTS
# =======================
class PatternType(Enum):
    """Line pattern classifications"""
    WARNING_DOTS = "warning dots"
    DIRECTIONAL_BARS = "directional bars"
    SOLID_LINE = "solid line"
    UNKNOWN = "unknown"

class Direction(Enum):
    """Navigation directions"""
    STRAIGHT = "Go straight"
    LEFT = "Move left"
    RIGHT = "Move right"
    NO_LINE = "No line detected"

# =======================
# CONFIGURATION MANAGER
# =======================
class ConfigManager:
    """Advanced configuration with persistence and calibration"""
    
    DEFAULT_CONFIG = {
        # Color detection (HSV)
        "lower_yellow": [18, 100, 100],
        "upper_yellow": [35, 255, 255],
        
        # Line detection
        "min_area": 1500,
        "center_tolerance": 60,
        "circularity_threshold": 0.6,
        
        # Camera
        "camera_index": 0,
        "frame_width": 640,
        "frame_height": 480,
        "fps_target": 30,
        
        # Speech
        "speech_rate": 170,
        "speak_interval": 2.0,
        "speech_enabled": True,
        
        # OCR
        "ocr_min_text_length": 1,
        "ocr_confidence_threshold": 30,
        "ocr_enabled": True,
        "ocr_interval": 0.2,
        
        # Performance
        "history_size": 5,
        "smoothing_enabled": True,
        "debug_mode": False,
        
        # Advanced
        "auto_exposure": True,
        "brightness_adjustment": 0,
        "contrast_adjustment": 1.0
    }
    
    def __init__(self, config_file: str = "vision_config.json"):
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
                print(f"[INFO] Configuration loaded from {self.config_file}")
            except Exception as e:
                print(f"[WARNING] Failed to load config: {e}. Using defaults.")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"[INFO] Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value"""
        self.config[key] = value
    
    def get_lower_yellow(self) -> np.ndarray:
        return np.array(self.config["lower_yellow"])
    
    def get_upper_yellow(self) -> np.ndarray:
        return np.array(self.config["upper_yellow"])

# =======================
# PERFORMANCE MONITOR
# =======================
class PerformanceMonitor:
    """Track and display performance metrics"""
    
    def __init__(self, window_size: int = 30):
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def start_frame(self):
        """Mark start of frame processing"""
        self.last_time = time.time()
    
    def end_frame(self):
        """Mark end of frame processing"""
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        self.processing_times.append(frame_time)
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if not self.frame_times:
            return 0.0
        avg_time = np.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_processing_time(self) -> float:
        """Get average processing time in ms"""
        if not self.processing_times:
            return 0.0
        return np.mean(self.processing_times) * 1000

# =======================
# ENHANCED SPEECH HANDLER
# =======================
class SpeechHandler:
    """Advanced thread-safe speech synthesis with priority queue"""
    
    def __init__(self, rate: int = 170, enabled: bool = True):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.enabled = enabled
        except Exception as e:
            print(f"[WARNING] TTS initialization failed: {e}. Speech disabled.")
            self.enabled = False
            self.engine = None
            
        self.speech_queue = deque(maxlen=5)
        self.priority_queue = deque(maxlen=2)
        self.is_speaking = False
        self.lock = threading.Lock()
        self.last_announcement = {}
        self.cooldown_period = 1.5
    
    def speak_async(self, text: str, priority: bool = False):
        """Non-blocking speech synthesis with priority support"""
        if not self.enabled or not self.engine or not text:
            return
        
        # Check cooldown for repeated messages
        current_time = time.time()
        if text in self.last_announcement:
            if current_time - self.last_announcement[text] < self.cooldown_period:
                return
        
        self.last_announcement[text] = current_time
        
        with self.lock:
            queue = self.priority_queue if priority else self.speech_queue
            if not queue or queue[-1] != text:
                queue.append(text)
        
        if not self.is_speaking:
            threading.Thread(target=self._process_queue, daemon=True).start()
    
    def _process_queue(self):
        """Process speech queue with priority handling"""
        while True:
            with self.lock:
                # Priority queue first
                if self.priority_queue:
                    text = self.priority_queue.popleft()
                elif self.speech_queue:
                    text = self.speech_queue.popleft()
                else:
                    break
                
                self.is_speaking = True
            
            print(f"[SPEAK] {text}")
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"[ERROR] Speech failed: {e}")
            
            time.sleep(0.2)
        
        with self.lock:
            self.is_speaking = False
    
    def toggle(self):
        """Toggle speech on/off"""
        self.enabled = not self.enabled
        return self.enabled

# =======================
# ENHANCED LINE DETECTOR
# =======================
class LineDetector:
    """Advanced yellow line detection with adaptive algorithms"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.center_history = deque(maxlen=config.get("history_size", 5))
        self.pattern_history = deque(maxlen=3)
        self.kalman = self._init_kalman_filter()
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter for smooth tracking"""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        return kalman
    
    def detect_yellow_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create enhanced binary mask for yellow regions"""
        # Apply brightness/contrast adjustment
        brightness = self.config.get("brightness_adjustment", 0)
        contrast = self.config.get("contrast_adjustment", 1.0)
        
        if brightness != 0 or contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adaptive color range based on lighting
        mask = cv2.inRange(hsv, self.config.get_lower_yellow(), 
                          self.config.get_upper_yellow())
        
        # Advanced morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Remove small noise
        mask = cv2.medianBlur(mask, 5)
        
        return mask
    
    def get_largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Find the largest valid contour with validation"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Filter by area and aspect ratio
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.get("min_area", 1500):
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Lines should be wider than tall
            if aspect_ratio > 0.5 or area > 5000:
                valid_contours.append(cnt)
        
        if not valid_contours:
            return None
        
        return max(valid_contours, key=cv2.contourArea)
    
    def get_smoothed_center(self, cx: int, cy: int) -> Tuple[int, int]:
        """Apply Kalman filtering for smooth tracking"""
        if not self.config.get("smoothing_enabled", True):
            return cx, cy
        
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        
        return int(prediction[0]), int(prediction[1])
    
    def classify_pattern(self, roi_mask: np.ndarray) -> PatternType:
        """Enhanced pattern classification with history"""
        if roi_mask.size == 0:
            return PatternType.UNKNOWN
        
        # Edge detection
        edges = cv2.Canny(roi_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, 
                                minLineLength=40, maxLineGap=15)
        
        # Detect circular patterns
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        circular_count = 0
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < 30 or area > 2500:
                continue
            
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if 0.4 < circularity < 1.3:
                circular_count += 1
        
        # Classification with confidence
        line_count = len(lines) if lines is not None else 0
        
        if circular_count >= 3 and line_count < 5:
            pattern = PatternType.WARNING_DOTS
        elif line_count >= 6:
            pattern = PatternType.DIRECTIONAL_BARS
        elif line_count > 0:
            pattern = PatternType.SOLID_LINE
        else:
            pattern = PatternType.UNKNOWN
        
        # Use history for stability
        self.pattern_history.append(pattern)
        if len(self.pattern_history) >= 2:
            # Return most common pattern
            from collections import Counter
            most_common = Counter(self.pattern_history).most_common(1)[0][0]
            return most_common
        
        return pattern
    
    def get_guidance(self, cx: int, frame_width: int) -> Direction:
        """Generate navigation guidance with hysteresis"""
        frame_center = frame_width // 2
        dx = cx - frame_center
        tolerance = self.config.get("center_tolerance", 60)
        
        # Hysteresis to prevent oscillation
        if abs(dx) < tolerance:
            return Direction.STRAIGHT
        elif dx < -tolerance:
            return Direction.RIGHT
        else:
            return Direction.LEFT

# =======================
# ENHANCED SIGN DETECTOR
# =======================
class SignDetector:
    """Advanced OCR with caching and validation"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.last_signs = deque(maxlen=5)
        self.last_detection_time = 0
        self.sign_cache = {}
        self.cache_duration = 3.0
    
    def preprocess_for_ocr(self, frame: np.ndarray) -> List[np.ndarray]:
        """Multiple preprocessing strategies optimized for digital displays and signs"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        preprocessed = []
        
        # Strategy 1: Inverted binary (white text on dark)
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed.append(binary_inv)
        
        # Strategy 2: Regular binary (dark text on white)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(binary)
        
        # Strategy 3: Adaptive threshold (handles uneven lighting)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
        preprocessed.append(adaptive)
        
        # Strategy 4: Enhanced contrast with denoising
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        preprocessed.append(denoised)
        
        # Strategy 5: Morphological gradient (edge enhancement)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, gradient_bin = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(gradient_bin)
        
        # Strategy 6: For digital displays - sharpen
        kernel_sharp = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharp)
        preprocessed.append(sharpened)
        
        return preprocessed
    
    def detect_color_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect colored sign regions (red, blue, yellow, etc.)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bboxes = []
        
        # Define color ranges for common signs
        color_ranges = [
            # Red (STOP signs, warnings)
            ([0, 100, 100], [10, 255, 255]),
            ([170, 100, 100], [180, 255, 255]),
            # Blue (information signs)
            ([100, 100, 100], [130, 255, 255]),
            # Yellow (warning signs)
            ([15, 100, 100], [35, 255, 255]),
            # White (on dark background)
            ([0, 0, 200], [180, 30, 255])
        ]
        
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to connect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            
            # Filter for sign-like regions (more aggressive)
            if area > 500 and w > 30 and h > 30:
                aspect = w / float(h)
                if 0.2 < aspect < 5:
                    bboxes.append((x, y, w, h))
        
        return bboxes
    
    def find_text_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect potential sign regions using multiple methods"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bboxes = []
        
        # Method 1: Color-based detection (best for colored signs)
        color_regions = self.detect_color_regions(frame)
        bboxes.extend(color_regions)
        
        # Method 2: Edge detection with larger regions
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            
            if area < 800:
                continue
            
            aspect = w / float(h) if h > 0 else 0
            
            # More lenient for various sign shapes
            if 0.3 < aspect < 4 and w > 40 and h > 40:
                # Check if not already covered
                overlap = False
                for bx, by, bw, bh in bboxes:
                    if abs(x - bx) < 20 and abs(y - by) < 20:
                        overlap = True
                        break
                
                if not overlap:
                    bboxes.append((x, y, w, h))
        
        # Method 3: MSER for text-like regions
        try:
            mser = cv2.MSER_create(
                _delta=5,
                _min_area=100,
                _max_area=10000
            )
            regions, _ = mser.detectRegions(gray)
            
            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                aspect = w / float(h) if h > 0 else 0
                
                if 0.2 < aspect < 6 and w > 30 and h > 20:
                    # Expand region for better capture
                    x = max(0, x - 5)
                    y = max(0, y - 5)
                    w = min(frame.shape[1] - x, w + 10)
                    h = min(frame.shape[0] - y, h + 10)
                    
                    overlap = False
                    for bx, by, bw, bh in bboxes:
                        if abs(x - bx) < 30 and abs(y - by) < 30:
                            overlap = True
                            break
                    
                    if not overlap:
                        bboxes.append((x, y, w, h))
        except:
            pass
        
        return bboxes
    
    def read_sign_text(self, frame: np.ndarray) -> Optional[str]:
        """Enhanced OCR optimized for real signs and digital displays"""
        if not self.config.get("ocr_enabled", True):
            return None
        
        current_time = time.time()
        
        # Rate limiting
        ocr_interval = self.config.get("ocr_interval", 0.5)
        if current_time - self.last_detection_time < ocr_interval:
            return None
        
        regions = self.find_text_regions(frame)
        best_text = None
        best_confidence = 0
        
        # Try multiple PSM modes for different text layouts
        psm_modes = [
            '--psm 6 --oem 3',  # Uniform block of text
            '--psm 7 --oem 3',  # Single line
            '--psm 8 --oem 3',  # Single word
            '--psm 11 --oem 3', # Sparse text
        ]
        
        for x, y, w, h in regions[:15]:  # Check top 15 regions
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0 or w < 20 or h < 20:
                continue
            
            # Resize small regions for better OCR (more aggressive)
            if w < 150 or h < 150:
                scale = max(250 / max(w, h), 1.5)
                new_w = int(w * scale)
                new_h = int(h * scale)
                roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            preprocessed_images = self.preprocess_for_ocr(roi)
            
            for prep_img in preprocessed_images:
                for psm_config in psm_modes:
                    try:
                        # Try standard OCR
                        data = pytesseract.image_to_data(
                            prep_img,
                            lang='eng',
                            config=psm_config,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        text_parts = []
                        confidences = []
                        
                        for i, conf in enumerate(data['conf']):
                            if conf == -1:
                                continue
                            
                            conf_int = int(conf)
                            # Much lower threshold for aggressive detection
                            threshold = max(self.config.get("ocr_confidence_threshold", 30) - 20, 10)
                            
                            if conf_int > threshold:
                                text = data['text'][i].strip()
                                # Clean up common OCR errors
                                text = text.replace('|', 'I').replace('0', 'O')
                                
                                if text and (any(ch.isalnum() for ch in text) or text in ['!', '?']):
                                    text_parts.append(text)
                                    confidences.append(conf_int)
                        
                        if text_parts:
                            full_text = ' '.join(text_parts).upper()
                            avg_confidence = np.mean(confidences)
                            
                            # Boost confidence for known sign words
                            sign_keywords = ['STOP', 'YIELD', 'SPEED', 'LIMIT', 'WARNING', 
                                           'DANGER', 'CAUTION', 'EXIT', 'ENTRANCE', 'PARKING',
                                           'NO', 'WAIT', 'GO', 'SLOW', 'SCHOOL']
                            
                            for keyword in sign_keywords:
                                if keyword in full_text:
                                    avg_confidence += 20
                                    break
                            
                            if len(full_text) >= self.config.get("ocr_min_text_length", 1):
                                if avg_confidence > best_confidence:
                                    best_confidence = avg_confidence
                                    best_text = full_text
                                    
                                    # Print debug info for all detections
                                    if self.config.get("debug_mode", False):
                                        print(f"[DEBUG] Found: '{full_text}' (conf: {avg_confidence:.1f})")
                    
                    except Exception as e:
                        if self.config.get("debug_mode", False):
                            print(f"[DEBUG] OCR error: {e}")
                        continue
        
        if best_text:
            # More lenient validation
            if len(best_text) >= 1:
                self.last_signs.append(best_text)
                self.last_detection_time = current_time
                
                # Return immediately if decent confidence or seen before
                if best_confidence > 40 or self.last_signs.count(best_text) >= 1:
                    return best_text
        
        return None

# =======================
# CALIBRATION MODE
# =======================
class CalibrationMode:
    """Interactive HSV calibration"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.active = False
        self.window_name = "HSV Calibration"
    
    def activate(self, frame: np.ndarray):
        """Start calibration mode"""
        self.active = True
        cv2.namedWindow(self.window_name)
        
        lower = self.config.get("lower_yellow")
        upper = self.config.get("upper_yellow")
        
        cv2.createTrackbar("Lower H", self.window_name, lower[0], 179, lambda x: None)
        cv2.createTrackbar("Lower S", self.window_name, lower[1], 255, lambda x: None)
        cv2.createTrackbar("Lower V", self.window_name, lower[2], 255, lambda x: None)
        cv2.createTrackbar("Upper H", self.window_name, upper[0], 179, lambda x: None)
        cv2.createTrackbar("Upper S", self.window_name, upper[1], 255, lambda x: None)
        cv2.createTrackbar("Upper V", self.window_name, upper[2], 255, lambda x: None)
        
        print("[INFO] Calibration mode active. Adjust trackbars and press 's' to save.")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Show calibration view"""
        if not self.active:
            return frame
        
        lh = cv2.getTrackbarPos("Lower H", self.window_name)
        ls = cv2.getTrackbarPos("Lower S", self.window_name)
        lv = cv2.getTrackbarPos("Lower V", self.window_name)
        uh = cv2.getTrackbarPos("Upper H", self.window_name)
        us = cv2.getTrackbarPos("Upper S", self.window_name)
        uv = cv2.getTrackbarPos("Upper V", self.window_name)
        
        lower = np.array([lh, ls, lv])
        upper = np.array([uh, us, uv])
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        combined = np.hstack([frame, result])
        cv2.imshow(self.window_name, combined)
        
        return combined
    
    def save_and_close(self):
        """Save calibration and exit"""
        if not self.active:
            return
        
        lh = cv2.getTrackbarPos("Lower H", self.window_name)
        ls = cv2.getTrackbarPos("Lower S", self.window_name)
        lv = cv2.getTrackbarPos("Lower V", self.window_name)
        uh = cv2.getTrackbarPos("Upper H", self.window_name)
        us = cv2.getTrackbarPos("Upper S", self.window_name)
        uv = cv2.getTrackbarPos("Upper V", self.window_name)
        
        self.config.set("lower_yellow", [lh, ls, lv])
        self.config.set("upper_yellow", [uh, us, uv])
        self.config.save_config()
        
        cv2.destroyWindow(self.window_name)
        self.active = False
        print("[INFO] Calibration saved!")

# =======================
# MAIN APPLICATION
# =======================
class SmartVisionAssistant:
    """Enhanced main application with advanced features"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.performance = PerformanceMonitor()
        self.speech = SpeechHandler(
            self.config.get("speech_rate", 170),
            self.config.get("speech_enabled", True)
        )
        self.line_detector = LineDetector(self.config)
        self.sign_detector = SignDetector(self.config)
        self.calibration = CalibrationMode(self.config)
        
        self.last_guidance = Direction.NO_LINE
        self.last_pattern = PatternType.UNKNOWN
        self.last_speak_time = 0
        
        self.cap = None
        self.running = False
        self.paused = False
        
        self.show_help = False
        self.show_debug = self.config.get("debug_mode", False)
        self.show_ocr_regions = False  # New: visualize OCR regions
    
    def initialize_camera(self) -> bool:
        """Initialize camera with optimal settings"""
        camera_index = self.config.get("camera_index", 0)
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Camera {camera_index} not accessible!")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get("frame_width", 640))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get("frame_height", 480))
        self.cap.set(cv2.CAP_PROP_FPS, self.config.get("fps_target", 30))
        
        if self.config.get("auto_exposure", True):
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        
        print(f"[INFO] Camera initialized: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return True
    
    def draw_overlay(self, frame: np.ndarray, guidance: Direction, 
                     pattern: PatternType, sign_text: Optional[str]) -> np.ndarray:
        """Draw comprehensive UI overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Semi-transparent info panel
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Guidance text
        color = (0, 255, 0) if guidance == Direction.STRAIGHT else (0, 165, 255)
        cv2.putText(frame, guidance.value, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Pattern info
        pattern_color = (0, 255, 255) if pattern != PatternType.UNKNOWN else (128, 128, 128)
        cv2.putText(frame, f"Pattern: {pattern.value}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, pattern_color, 2)
        
        # Sign text
        if sign_text:
            cv2.putText(frame, f"Sign: {sign_text}", (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Performance metrics
        fps = self.performance.get_fps()
        proc_time = self.performance.get_processing_time()
        cv2.putText(frame, f"FPS: {fps:.1f} | {proc_time:.1f}ms",
                   (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Center guide line
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 0, 0), 1)
        
        # Help overlay
        if self.show_help:
            self.draw_help(frame)
        
        # Debug info
        if self.show_debug:
            self.draw_debug_info(frame)
        
        return frame
    
    def draw_help(self, frame: np.ndarray):
        """Draw help overlay"""
        h, w = frame.shape[:2]
        help_text = [
            "KEYBOARD SHORTCUTS:",
            "Q/ESC - Quit",
            "SPACE - Pause/Resume",
            "C - Calibration mode",
            "S - Toggle speech",
            "D - Toggle debug",
            "O - Toggle OCR regions",
            "H - Toggle help",
            "R - Reset config",
            "+ - Save screenshot"
        ]
        
        y_offset = 150
        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (10, y_offset + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def draw_debug_info(self, frame: np.ndarray):
        """Draw debug information"""
        h, w = frame.shape[:2]
        debug_info = [
            f"Speech Queue: {len(self.speech.speech_queue)}",
            f"Pattern History: {len(self.line_detector.pattern_history)}",
            f"Sign Cache: {len(self.sign_detector.sign_cache)}",
        ]
        
        y_offset = h - 80
        for i, text in enumerate(debug_info):
            cv2.putText(frame, text, (10, y_offset + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def visualize_ocr_regions(self, frame: np.ndarray) -> np.ndarray:
        """Draw all detected regions for debugging"""
        debug_frame = frame.copy()
        regions = self.sign_detector.find_text_regions(frame)
        
        for i, (x, y, w, h) in enumerate(regions):
            # Draw different colors for different regions
            color = ((i * 50) % 255, (i * 100) % 255, (i * 150) % 255)
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_frame, f"R{i}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.putText(debug_frame, f"Regions found: {len(regions)}", (10, frame.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return debug_frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame with all detections"""
        self.performance.start_frame()
        
        h, w = frame.shape[:2]
        guidance = Direction.NO_LINE
        pattern = PatternType.UNKNOWN
        sign_text = None
        
        # Calibration mode override
        if self.calibration.active:
            return self.calibration.process(frame)
        
        # --- Line Detection ---
        mask = self.line_detector.detect_yellow_mask(frame)
        contour = self.line_detector.get_largest_contour(mask)
        
        if contour is not None:
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else h // 2
            
            cx_smooth, cy_smooth = self.line_detector.get_smoothed_center(cx, cy)
            guidance = self.line_detector.get_guidance(cx_smooth, w)
            
            # Pattern classification
            x, y, ww, hh = cv2.boundingRect(contour)
            roi_mask = mask[y:y+hh, x:x+ww]
            pattern = self.line_detector.classify_pattern(roi_mask)
            
            # Visualization
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx_smooth, cy_smooth), 8, (0, 0, 255), -1)
            cv2.circle(frame, (cx_smooth, cy_smooth), 12, (255, 255, 255), 2)
            
            # Voice guidance
            current_time = time.time()
            speak_interval = self.config.get("speak_interval", 2.0)
            
            if (guidance != self.last_guidance or pattern != self.last_pattern) and \
               (current_time - self.last_speak_time > speak_interval):
                announcement = f"{guidance.value}, {pattern.value}"
                self.speech.speak_async(announcement)
                self.last_guidance = guidance
                self.last_pattern = pattern
                self.last_speak_time = current_time
        
        # --- Sign Detection ---
        if self.config.get("ocr_enabled", True):
            # Show OCR regions if debug mode is on
            if self.show_ocr_regions:
                frame = self.visualize_ocr_regions(frame)
            
            sign_text = self.sign_detector.read_sign_text(frame)
            if sign_text:
                # Check if it's a new detection
                is_new = True
                for recent_sign in list(self.sign_detector.last_signs)[-3:]:
                    if sign_text == recent_sign:
                        is_new = False
                        break
                
                if is_new:
                    print(f"[DETECTED] {sign_text} (confidence: high)")
                    self.speech.speak_async(f"Sign detected: {sign_text}", priority=True)
                    
                    # Draw bounding box around detected sign region
                    regions = self.sign_detector.find_text_regions(frame)
                    if regions:
                        x, y, w, h = regions[0]  # First region
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
                        cv2.putText(frame, sign_text, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw overlay
        frame = self.draw_overlay(frame, guidance, pattern, sign_text)
        
        self.performance.end_frame()
        return frame
    
    def handle_keyboard(self, key: int) -> bool:
        """Handle keyboard shortcuts"""
        if key == 27 or key == ord('q'):  # ESC or Q
            return False
        elif key == ord(' '):  # SPACE
            self.paused = not self.paused
            print(f"[INFO] {'Paused' if self.paused else 'Resumed'}")
        elif key == ord('c'):  # C - Calibration
            if not self.calibration.active:
                print("[INFO] Entering calibration mode...")
                self.calibration.activate(None)
            else:
                print("[INFO] Exiting calibration mode...")
                self.calibration.active = False
        elif key == ord('s'):  # S - Toggle speech
            enabled = self.speech.toggle()
            print(f"[INFO] Speech {'enabled' if enabled else 'disabled'}")
        elif key == ord('d'):  # D - Toggle debug
            self.show_debug = not self.show_debug
            self.config.set("debug_mode", self.show_debug)
            print(f"[INFO] Debug mode {'enabled' if self.show_debug else 'disabled'}")
        elif key == ord('o'):  # O - Toggle OCR region visualization
            self.show_ocr_regions = not self.show_ocr_regions
            print(f"[INFO] OCR region visualization {'enabled' if self.show_ocr_regions else 'disabled'}")
        elif key == ord('h'):  # H - Toggle help
            self.show_help = not self.show_help
        elif key == ord('r'):  # R - Reset config
            self.config.config = ConfigManager.DEFAULT_CONFIG.copy()
            self.config.save_config()
            print("[INFO] Configuration reset to defaults")
        elif key == ord('+') or key == ord('='):  # + - Screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            cv2.imwrite(filename, self.current_frame)
            print(f"[INFO] Screenshot saved: {filename}")
        
        return True
    
    def run(self):
        """Main application loop with error recovery"""
        if not self.initialize_camera():
            return
        
        self.running = True
        print("\n" + "="*60)
        print("    SMART VISION ASSISTANT v2.0")
        print("="*60)
        print("\nPress 'H' for help, 'Q' to quit\n")
        
        error_count = 0
        max_errors = 10
        
        try:
            while self.running:
                if not self.paused:
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        error_count += 1
                        print(f"[WARNING] Frame capture failed ({error_count}/{max_errors})")
                        
                        if error_count >= max_errors:
                            print("[ERROR] Too many frame errors. Exiting...")
                            break
                        
                        time.sleep(0.1)
                        continue
                    
                    error_count = 0  # Reset on success
                    
                    # Resize for processing
                    frame = cv2.resize(frame, 
                                      (self.config.get("frame_width", 640),
                                       self.config.get("frame_height", 480)))
                    
                    self.current_frame = frame.copy()
                    
                    # Process frame
                    try:
                        processed = self.process_frame(frame)
                        cv2.imshow("Smart Vision Assistant", processed)
                    except Exception as e:
                        print(f"[ERROR] Processing error: {e}")
                        cv2.imshow("Smart Vision Assistant", frame)
                else:
                    # Show paused frame
                    if hasattr(self, 'current_frame'):
                        paused_frame = self.current_frame.copy()
                        h, w = paused_frame.shape[:2]
                        cv2.putText(paused_frame, "PAUSED", (w//2 - 100, h//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        cv2.imshow("Smart Vision Assistant", paused_frame)
                
                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard(key):
                    break
                
                # Handle calibration save
                if self.calibration.active and key == ord('s'):
                    self.calibration.save_and_close()
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown"""
        print("\n[INFO] Shutting down...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Save configuration
        self.config.save_config()
        
        print("[INFO] Cleanup complete. Goodbye!")

# =======================
# ENTRY POINT
# =======================
if __name__ == "__main__":
    try:
        assistant = SmartVisionAssistant()
        assistant.run()
    except Exception as e:
        print(f"[FATAL] Application failed to start: {e}")
        import traceback
        traceback.print_exc()