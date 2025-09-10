from __future__ import annotations
import ctypes
import time
import re
import webbrowser
import subprocess
import os
import glob
from typing import Generator, Iterable, Optional, List, Tuple, Dict
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, Future
import tkinter as tk
from tkinter import ttk
import math

import pyautogui
import pygetwindow as gw
import pytesseract
import cv2
import numpy as np
import win32gui
import win32con

try:
    # deep-translator is lightweight and reliable
    from deep_translator import GoogleTranslator
except Exception as _import_error:  # pragma: no cover
    GoogleTranslator = None  # type: ignore


# --- Configuration ---
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SILENCE_SEC: float = 0.3  # Further reduced for more responsive capture
CAPTURE_INTERVAL_SEC: float = 0.2  # Further reduced for faster capture
MAX_WORDS_DISPLAY: int = 40  # Maximum number of words to display in caption window
IGNORE_SUBSTRINGS: List[str] = [
    "ready to show live captions",
    "ready to show live captions in english",
]
TEST_MODE = True  # Set to True for immediate capture without stability checks

# Point pytesseract to local tesseract (if installed in default path)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


# --- Utilities reused from the existing app ---
def send_hotkey_open_live_captions() -> None:
    """Trigger Windows Live Captions with Win+Ctrl+L."""
    user32 = ctypes.windll.user32
    key_events = [
        (0x5B, 0),  # Win down
        (0x11, 0),  # Ctrl down
        (0x4C, 0),  # L down
        (0x4C, 2),  # L up
        (0x11, 2),  # Ctrl up
        (0x5B, 2),  # Win up
    ]
    for virtual_key, flag in key_events:
        user32.keybd_event(virtual_key, 0, flag, 0)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def canonicalize_text(text: str) -> str:
    """Aggressive canonical form for duplicate detection: letters+digits only, single spaces, lowercase."""
    lowered = text.lower()
    # Replace non-alphanumeric with space, collapse spaces
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_captured_text(text: str) -> str:
    """Clean captured text by removing repeated blocks and normalizing."""
    if not text or not text.strip():
        return ""
    
    # First, normalize the text
    normalized = normalize_text(text)
    
    # Split into words and remove consecutive duplicates
    words = re.findall(r"[A-Za-z0-9']+", normalized)
    words = [w for w in words if w.strip()]
    
    if not words:
        return ""
    
    # Remove consecutive duplicates
    cleaned_words = [words[0]]
    for w in words[1:]:
        if w != cleaned_words[-1]:
            cleaned_words.append(w)
    words = cleaned_words
    
    # Remove repeated blocks (like "word word word word" -> "word word")
    n = len(words)
    if n >= 4:
        for k in range(1, min(n // 2 + 1, 10)):  # Limit to reasonable block sizes
            if n % k != 0:
                continue
            block = words[:k]
            if block * (n // k) == words:
                words = block
                break
    
    # Join back and do final cleanup
    cleaned = " ".join(words)
    
    # Remove very long repeated phrases (more aggressive)
    cleaned = re.sub(r'\b(\w+(?:\s+\w+){1,5})\s+\1\b', r'\1', cleaned)
    
    # Remove extremely repetitive patterns
    cleaned = re.sub(r'\b(\w+)\s+\1\s+\1\b', r'\1', cleaned)
    
    # Final cleanup - remove any remaining excessive repetition
    words_final = cleaned.split()
    if len(words_final) > 20:  # If text is too long, try to extract meaningful parts
        # Look for sentence boundaries and take the last complete sentence
        for i in range(len(words_final) - 1, max(0, len(words_final) - 10), -1):
            if words_final[i] in ['so', 'up', 'down', 'out', 'in', 'on', 'at', 'to', 'for', 'with']:
                cleaned = " ".join(words_final[i:])
                break
    
    return cleaned.strip()


def find_live_captions_window() -> Optional[gw.Win32Window]:
    return next((w for w in gw.getWindowsWithTitle("Live Captions")), None)


def launch_live_captions_exe() -> bool:
    """
    Launch LiveCaptions.exe and position the window.
    Returns True if successful, False otherwise.
    """
    try:
        # Launch LiveCaptions.exe
        subprocess.Popen(['LiveCaptions.exe'])
        
        # Wait for application to load
        time.sleep(3)
        
        # Find the Live Captions window
        hwnd = win32gui.FindWindow(None, "Live Captions")
        
        if hwnd:
            # Move and resize window to position (100, 100) with size 710x120
            win32gui.MoveWindow(hwnd, 100, 100, 710, 120, True)
            print("LiveCaptions.exe launched and window repositioned successfully!")
            return True
        else:
            print("Live Captions window not found!")
            return False
            
    except Exception as e:
        print(f"Error launching LiveCaptions.exe: {e}")
        return False


def cleanup_old_log_files(max_files: int = 5) -> None:
    """
    Clean up old log files, keeping only the most recent ones.
    
    Args:
        max_files: Maximum number of log files to keep (default: 5)
    """
    try:
        # Get all caption log files
        caption_logs = glob.glob("captions_log_*.txt")
        # Get all translated words log files
        words_logs = glob.glob("translated_words_*.txt")
        
        # Sort by modification time (newest first)
        caption_logs.sort(key=os.path.getmtime, reverse=True)
        words_logs.sort(key=os.path.getmtime, reverse=True)
        
        # Remove excess caption log files
        if len(caption_logs) > max_files:
            for old_file in caption_logs[max_files:]:
                try:
                    os.remove(old_file)
                    print(f"Removed old caption log: {old_file}")
                except Exception as e:
                    print(f"Error removing {old_file}: {e}")
        
        # Remove excess translated words log files
        if len(words_logs) > max_files:
            for old_file in words_logs[max_files:]:
                try:
                    os.remove(old_file)
                    print(f"Removed old words log: {old_file}")
                except Exception as e:
                    print(f"Error removing {old_file}: {e}")
                    
    except Exception as e:
        print(f"Error during log cleanup: {e}")


class WindowManager:
    """Manages Live Captions window positioning and movement."""
    
    def __init__(self, live_captions_window: gw.Win32Window):
        self.window = live_captions_window
        self.hwnd = None
        
    def get_hwnd(self) -> Optional[int]:
        """Get the window handle for the Live Captions window."""
        if not self.hwnd and self.window:
            try:
                self.hwnd = self.window._hWnd
            except:
                # Fallback: find window by title
                self.hwnd = win32gui.FindWindow(None, "Live Captions")
        return self.hwnd
    
    def move_window(self, x: int, y: int) -> bool:
        """Move the Live Captions window to the specified position."""
        try:
            hwnd = self.get_hwnd()
            if not hwnd:
                return False
                
            # Get current window size
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            # Move the window
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, x, y, width, height, 
                                win32con.SWP_SHOWWINDOW)
            
            # Update the pygetwindow object
            self.window.left = x
            self.window.top = y
            
            return True
        except Exception as e:
            print(f"Error moving window: {e}")
            return False
    
    def get_window_position(self) -> Tuple[int, int]:
        """Get current window position."""
        try:
            hwnd = self.get_hwnd()
            if hwnd:
                rect = win32gui.GetWindowRect(hwnd)
                return (rect[0], rect[1])
        except Exception as e:
            print(f"Error getting window position: {e}")
        return (self.window.left, self.window.top) if self.window else (0, 0)
    
    def get_window_size(self) -> Tuple[int, int]:
        """Get current window size."""
        try:
            hwnd = self.get_hwnd()
            if hwnd:
                rect = win32gui.GetWindowRect(hwnd)
                return (rect[2] - rect[0], rect[3] - rect[1])
        except Exception as e:
            print(f"Error getting window size: {e}")
        return (self.window.width, self.window.height) if self.window else (0, 0)


# --- Core functionality ---
def capture_live_captions(
    silence_seconds: float = SILENCE_SEC,
    capture_interval_seconds: float = CAPTURE_INTERVAL_SEC,
    ignore_substrings: Iterable[str] = IGNORE_SUBSTRINGS,
) -> Generator[str, None, None]:
    """
    Continuously OCR the "Live Captions" window and yield English text chunks.

    Logic: capture current text state, wait for it to be complete and stable, then emit only the final version.
    """
    # Ensure Live Captions window is visible
    send_hotkey_open_live_captions()
    time.sleep(3)

    window = find_live_captions_window()
    if not window:
        raise RuntimeError("Live Captions window not found. Make sure it's available on this system.")

    print(f"DEBUG: Found Live Captions window at {window.left}, {window.top}, {window.width}x{window.height}")
    previous_caption: str = ""
    last_activity_time = time.time()
    stable_text_count = 0  # Count how many times we've seen the same text
    last_yielded_text = ""  # Track what we last yielded to avoid duplicates
    caption_start_time = 0  # Track when current caption started

    while True:
        try:
            # Capture only the Live Captions window region for speed and accuracy
            screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
            raw_text = pytesseract.image_to_string(screenshot, lang="eng").strip()

            if raw_text:
                current_text = normalize_text(raw_text)
                
                # Debug: print what we're capturing
                if current_text != previous_caption:
                    print(f"DEBUG: Captured new text: '{current_text[:100]}{'...' if len(current_text) > 100 else ''}'")
                
                # Check if text has changed significantly
                if current_text != previous_caption:
                    # Text has changed, reset stability counter and start time
                    stable_text_count = 0
                    last_activity_time = time.time()
                    caption_start_time = time.time()
                    previous_caption = current_text
                    print(f"DEBUG: Text changed, reset counters. Stable count: {stable_text_count}")
                else:
                    # Same text, increment stability counter
                    stable_text_count += 1
                    last_activity_time = time.time()
                    print(f"DEBUG: Same text, stable count: {stable_text_count}, time since start: {now - caption_start_time:.1f}s")

            now = time.time()
            
            # Test mode: immediate capture for debugging
            if TEST_MODE and previous_caption and previous_caption != last_yielded_text:
                # Clean the text before yielding
                cleaned = clean_captured_text(previous_caption)
                if cleaned and len(cleaned.split()) > 0 and cleaned != last_yielded_text:
                    print(f"DEBUG: TEST MODE - yielding caption: '{cleaned}'")
                    yield cleaned
                    last_yielded_text = cleaned
                    # Reset after yielding to avoid duplicates
                    previous_caption = ""
                    stable_text_count = 0
                    caption_start_time = 0
            # Normal mode: only emit text if it's been stable for a while AND we've had silence for the required time
            elif (previous_caption and 
                previous_caption != last_yielded_text and
                stable_text_count >= 2 and  # Text must be stable for at least 2 captures (reduced from 3)
                (now - last_activity_time >= silence_seconds) and
                (now - caption_start_time >= 0.5) and  # Caption must have been building for at least 0.5 seconds (reduced from 1.0)
                all(substr not in previous_caption for substr in ignore_substrings)):
                
                # Clean the text before yielding
                cleaned = clean_captured_text(previous_caption)
                if cleaned and len(cleaned.split()) > 0 and cleaned != last_yielded_text:
                    print(f"DEBUG: FINAL CAPTION - yielding: '{cleaned}'")
                    yield cleaned
                    last_yielded_text = cleaned
                    # Reset after yielding to avoid duplicates
                    previous_caption = ""
                    stable_text_count = 0
                    caption_start_time = 0
            
            # Fallback: if we've had text for a while but it's not meeting the strict requirements, still emit it
            elif (previous_caption and 
                  previous_caption != last_yielded_text and
                  stable_text_count >= 1 and
                  (now - last_activity_time >= silence_seconds * 1.5) and  # Wait a bit longer
                  all(substr not in previous_caption for substr in ignore_substrings)):
                
                # Clean the text before yielding
                cleaned = clean_captured_text(previous_caption)
                if cleaned and len(cleaned.split()) > 0 and cleaned != last_yielded_text:
                    print(f"DEBUG: FALLBACK CAPTION - yielding: '{cleaned}'")
                    yield cleaned
                    last_yielded_text = cleaned
                    # Reset after yielding to avoid duplicates
                    previous_caption = ""
                    stable_text_count = 0
                    caption_start_time = 0

        except Exception as e:
            print(f"DEBUG: Error in capture loop: {e}")
            
        time.sleep(capture_interval_seconds)


def translate_text(text: str, target_language: str = "ru") -> Optional[str]:
    """
    Translate text to the target language using deep-translator's GoogleTranslator.
    Returns None if translation fails (e.g., no internet or API block).
    """
    if not text:
        return ""
    if GoogleTranslator is None:
        print("GoogleTranslator not available")
        return None
    try:
        # Using auto source detection is robust for English input
        translator = GoogleTranslator(source="auto", target=target_language)
        result = translator.translate(text)
        if result and result.strip():
            print(f"Translation successful: '{text}' -> '{result}'")
            return result.strip()
        else:
            print(f"Translation returned empty result for: '{text}'")
            return None
    except Exception as e:
        print(f"Translation failed for '{text}': {e}")
        # Network errors or service unavailability
        return None


def simple_translate_fallback(text: str, target_language: str = "ru") -> str:
    """Simple fallback translation using basic word mapping."""
    if target_language == "ru":
        # Basic English to Russian word mapping
        word_map = {
            "hello": "привет", "world": "мир", "good": "хорошо", "bad": "плохо",
            "yes": "да", "no": "нет", "thank": "спасибо", "you": "ты",
            "the": "это", "and": "и", "or": "или", "but": "но",
            "guy": "парень", "knows": "знает", "about": "о", "this": "это",
            "shit": "дерьмо", "fucking": "чертов", "need": "нужно", "we": "мы",
            "scheduler": "планировщик", "wnfl": "wnfl", "wnel": "wnel",
            "hey": "эй", "got": "получил", "a": "а", "normal": "нормальная", "team": "команда",
            "today": "сегодня", "well": "хорошо", "that": "то", "that's": "это", "fire": "огонь",
            "otherwise": "иначе", "you": "ты", "bite": "кусать", "activeness": "активность",
            "what": "что", "the": "это", "fuck": "блять", "will": "будет", "mother": "мать", "fucker": "уебок",
            "ok": "ок", "let": "пусть", "me": "мне", "get": "получить", "that": "то", "yeah": "да",
            "exactly": "точно", "on": "на", "want": "хочу", "six": "шесть", "chicken": "курица",
            "wings": "крылья", "can": "могу", "three": "три", "more": "больше", "orders": "заказы",
            "at": "в", "all": "все", "i": "я", "want": "хочу", "get": "получить",
            "annoying": "раздражающий", "man": "мужчина", "is": "есть", "are": "есть", 
            "was": "был", "were": "были", "be": "быть", "have": "иметь", "has": "имеет", 
            "had": "имел", "do": "делать", "does": "делает", "did": "делал", "would": "бы", 
            "could": "мог", "should": "должен", "may": "может", "might": "может", "must": "должен",
            "cash": "деньги", "away": "прочь", "our": "наш", "hello": "привет", "hi": "привет",
            "how": "как", "doing": "делаешь", "going": "идешь", "coming": "приходишь", 
            "here": "здесь", "there": "там", "where": "где", "when": "когда", "why": "почему", 
            "who": "кто", "which": "который", "whose": "чей", "leaving": "уходящий", "see": "видеть",
            "to": "к", "not": "не", "just": "просто", "only": "только", "also": "также",
            "too": "тоже", "very": "очень", "really": "действительно", "actually": "на самом деле",
            "now": "сейчас", "then": "тогда", "before": "до", "after": "после", "always": "всегда",
            "never": "никогда", "sometimes": "иногда", "often": "часто", "usually": "обычно",
            "take": "взять", "go": "идти", "little": "маленький", "tour": "тур", "show": "показать",
            "some": "некоторые", "spots": "места", "sure": "конечно", "i'm": "я", "of": "из"
        }
        words = text.lower().split()
        translated = [word_map.get(word, word) for word in words]
        return " ".join(translated)
    return text


def translate_sentence_with_highlighted_word(sentence: str, clicked_word: str, target_language: str = "ru") -> str:
    """Translate full sentence and highlight the clicked word in the translation."""
    if target_language == "ru":
        word_map = {
            "hello": "привет", "world": "мир", "good": "хорошо", "bad": "плохо",
            "yes": "да", "no": "нет", "thank": "спасибо", "you": "ты",
            "the": "это", "and": "и", "or": "или", "but": "но",
            "guy": "парень", "knows": "знает", "about": "о", "this": "это",
            "shit": "дерьмо", "fucking": "чертов", "need": "нужно", "we": "мы",
            "scheduler": "планировщик", "wnfl": "wnfl", "wnel": "wnel",
            "hey": "эй", "got": "получил", "a": "а", "normal": "нормальная", "team": "команда",
            "today": "сегодня", "well": "хорошо", "that": "то", "that's": "это", "fire": "огонь",
            "otherwise": "иначе", "you": "ты", "bite": "кусать", "activeness": "активность",
            "what": "что", "the": "это", "fuck": "блять", "will": "будет", "mother": "мать", "fucker": "уебок",
            "ok": "ок", "let": "пусть", "me": "мне", "get": "получить", "that": "то", "yeah": "да",
            "exactly": "точно", "on": "на", "want": "хочу", "six": "шесть", "chicken": "курица",
            "wings": "крылья", "can": "могу", "three": "три", "more": "больше", "orders": "заказы",
            "at": "в", "all": "все", "i": "я", "want": "хочу", "get": "получить",
            "annoying": "раздражающий", "man": "мужчина", "is": "есть", "are": "есть", 
            "was": "был", "were": "были", "be": "быть", "have": "иметь", "has": "имеет", 
            "had": "имел", "do": "делать", "does": "делает", "did": "делал", "would": "бы", 
            "could": "мог", "should": "должен", "may": "может", "might": "может", "must": "должен",
            "cash": "деньги", "away": "прочь", "our": "наш", "hello": "привет", "hi": "привет",
            "how": "как", "doing": "делаешь", "going": "идешь", "coming": "приходишь", 
            "here": "здесь", "there": "там", "where": "где", "when": "когда", "why": "почему", 
            "who": "кто", "which": "который", "whose": "чей", "leaving": "уходящий", "see": "видеть",
            "to": "к", "not": "не", "just": "просто", "only": "только", "also": "также",
            "too": "тоже", "very": "очень", "really": "действительно", "actually": "на самом деле",
            "now": "сейчас", "then": "тогда", "before": "до", "after": "после", "always": "всегда",
            "never": "никогда", "sometimes": "иногда", "often": "часто", "usually": "обычно",
            "take": "взять", "go": "идти", "little": "маленький", "tour": "тур", "show": "показать",
            "some": "некоторые", "spots": "места", "sure": "конечно", "i'm": "я", "of": "из"
        }
        
        words = sentence.lower().split()
        translated_words = []
        clicked_word_lower = clicked_word.lower()
        
        for word in words:
            translated_word = word_map.get(word, word)
            if word == clicked_word_lower:
                translated_words.append(f"**{translated_word}**")  # Highlight the clicked word
            else:
                translated_words.append(translated_word)
        
        return " ".join(translated_words)
    return sentence


def run_realtime_caption_translation(target_language: str = "ru", log_to_file: Optional[str] = None) -> None:
    """
    Start the capture loop and print English + translated text in real time.
    Optionally, append the pairs to a log file.
    """
    # Clean up old log files before starting
    cleanup_old_log_files()
    
    log_file_handle = None
    if log_to_file:
        log_file_handle = open(log_to_file, "a", encoding="utf-8")

    try:
        for english_chunk in capture_live_captions():
            translated = translate_text(english_chunk, target_language=target_language)

            # Prepare output lines
            en_line = f"EN: {english_chunk}"
            if translated is None:
                tr_line = f"{target_language.upper()}: [translation unavailable]"
            else:
                tr_line = f"{target_language.upper()}: {translated}"

            print(en_line, flush=True)
            print(tr_line, flush=True)

            if log_file_handle:
                log_file_handle.write(en_line + "\n")
                log_file_handle.write(tr_line + "\n")
                log_file_handle.flush()
    finally:
        if log_file_handle:
            log_file_handle.close()


class CaptionCaptureThread(threading.Thread):
    """Background thread to capture captions and push them to a queue."""

    def __init__(self, output_queue: "queue.Queue[str]", silence_seconds: float = SILENCE_SEC, capture_interval_seconds: float = CAPTURE_INTERVAL_SEC) -> None:
        super().__init__(daemon=True)
        self.output_queue = output_queue
        self.silence_seconds = silence_seconds
        self.capture_interval_seconds = capture_interval_seconds
        self._stop_event = threading.Event()

    def run(self) -> None:
        try:
            for sentence in capture_live_captions(
                silence_seconds=self.silence_seconds,
                capture_interval_seconds=self.capture_interval_seconds,
            ):
                if self._stop_event.is_set():
                    break
                self.output_queue.put(sentence)
        except Exception as exc:  # Communicate fatal errors to UI
            self.output_queue.put(f"__ERROR__:{exc}")

    def stop(self) -> None:
        self._stop_event.set()


class WordDetector:
    """Computer vision-based word detection and bounding box management."""
    
    def __init__(self, live_captions_window: gw.Win32Window, window_manager: WindowManager):
        self.window = live_captions_window
        self.window_manager = window_manager
        self.word_boxes: List[Dict] = []  # List of word bounding boxes with metadata
        self.selected_word: Optional[str] = None
        self.hovered_word: Optional[str] = None
        self.overlay_window: Optional[tk.Toplevel] = None
        self.canvas: Optional[tk.Canvas] = None
        self.is_active = False
        
    def detect_words(self, screenshot: np.ndarray) -> List[Dict]:
        """Detect words in the Live Captions window and return bounding boxes."""
        try:
            print("WordDetector - Starting word detection...")
            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            print(f"WordDetector - Image shape: {gray.shape}")
            
            # Use pytesseract to get word-level data
            print("WordDetector - Running OCR...")
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            print(f"WordDetector - OCR returned {len(data['text'])} text elements")
            
            word_boxes = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                # Only process words with reasonable confidence and non-empty text
                if conf > 30 and text and len(text) > 0:
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    
                    word_boxes.append({
                        'text': text,
                        'bbox': (x, y, w, h),
                        'confidence': conf,
                        'selected': False
                    })
            
            self.word_boxes = word_boxes
            return word_boxes
            
        except Exception as e:
            print(f"Error detecting words: {e}")
            return []
    
    def create_overlay_window(self) -> None:
        """Create a transparent overlay window for interactive bounding boxes."""
        if not self.window:
            print("WordDetector - No window reference, cannot create overlay")
            return
            
        print(f"WordDetector - Creating overlay window at ({self.window.left}, {self.window.top}) size {self.window.width}x{self.window.height}")
        self.overlay_window = tk.Toplevel()
        self.overlay_window.title("Live Captions - Word Overlay")
        self.overlay_window.geometry(f"{self.window.width}x{self.window.height}+{self.window.left}+{self.window.top}")
        
        # Make window semi-transparent and always on top
        self.overlay_window.attributes('-alpha', 0.3)  # Semi-transparent so bounding boxes are visible
        self.overlay_window.attributes('-topmost', True)
        self.overlay_window.overrideredirect(True)
        
        # Create canvas for drawing bounding boxes
        self.canvas = tk.Canvas(
            self.overlay_window,
            width=self.window.width,
            height=self.window.height,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Bind events
        self.canvas.bind("<Button-1>", self._on_word_click)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Leave>", self._on_mouse_leave)
        
    def draw_bounding_boxes(self) -> None:
        """Draw bounding boxes around detected words with hover and selection effects."""
        if not self.canvas or not self.word_boxes:
            return
        
        # Check if canvas still exists (not destroyed)
        try:
            self.canvas.winfo_exists()
        except tk.TclError:
            # Canvas was destroyed, skip drawing
            return
            
        # Clear previous drawings
        self.canvas.delete("all")
        
        for word_data in self.word_boxes:
            x, y, w, h = word_data['bbox']
            text = word_data['text']
            selected = word_data.get('selected', False)
            hovered = (self.hovered_word == text)
            
            # Choose colors based on state
            if selected:
                outline_color = 'red'
                fill_color = 'red'
                fill_alpha = 0.3
            elif hovered:
                outline_color = 'orange'
                fill_color = 'orange'
                fill_alpha = 0.2
            else:
                outline_color = 'blue'
                fill_color = 'blue'
                fill_alpha = 0.1
            
            # Draw bounding box with fill
            self.canvas.create_rectangle(
                x, y, x + w, y + h,
                outline=outline_color,
                width=3 if selected else 2,
                fill=fill_color,
                tags=f"word_{text}"
            )
            
            # Draw word text for better visibility
            # Text labels removed for cleaner display
    
    def _on_word_click(self, event) -> None:
        """Handle click events on bounding boxes."""
        print(f"WordDetector - Word click detected at ({event.x}, {event.y})")
        if not self.word_boxes:
            print("WordDetector - No word boxes available")
            return
            
        click_x, click_y = event.x, event.y
        
        # Find which word was clicked
        for word_data in self.word_boxes:
            x, y, w, h = word_data['bbox']
            print(f"Checking word '{word_data['text']}' at ({x}, {y}, {w}, {h})")
            if x <= click_x <= x + w and y <= click_y <= y + h:
                print(f"Word '{word_data['text']}' clicked!")
                # Deselect all other words
                for other_word in self.word_boxes:
                    other_word['selected'] = False
                
                # Select this word
                word_data['selected'] = True
                self.selected_word = word_data['text']
                
                # Redraw bounding boxes
                self.draw_bounding_boxes()
                
                # Trigger translation
                if hasattr(self, 'on_word_selected'):
                    print(f"Calling on_word_selected for '{word_data['text']}'")
                    self.on_word_selected(word_data['text'])
                
                break
    
    def _on_mouse_motion(self, event) -> None:
        """Handle mouse motion events for hover effects."""
        if not self.word_boxes:
            return
            
        # Find which word is being hovered
        hovered_word = None
        for word_data in self.word_boxes:
            x, y, w, h = word_data['bbox']
            if x <= event.x <= x + w and y <= event.y <= y + h:
                hovered_word = word_data['text']
                break
        
        # Update hover state and redraw if changed
        if self.hovered_word != hovered_word:
            self.hovered_word = hovered_word
            self.draw_bounding_boxes()
    
    def _on_mouse_leave(self, event) -> None:
        """Handle mouse leave events."""
        if self.hovered_word:
            self.hovered_word = None
            self.draw_bounding_boxes()
    
    def update_window_position(self) -> None:
        """Update overlay window position to match Live Captions window."""
        if self.overlay_window and self.window_manager:
            try:
                current_x, current_y = self.window_manager.get_window_position()
                current_width, current_height = self.window_manager.get_window_size()
                
                # Update overlay window geometry
                self.overlay_window.geometry(f"{current_width}x{current_height}+{current_x}+{current_y}")
                if self.canvas:
                    self.canvas.configure(width=current_width, height=current_height)
                
                # Update the window object coordinates
                self.window.left = current_x
                self.window.top = current_y
                self.window.width = current_width
                self.window.height = current_height
                
            except Exception as e:
                print(f"Error updating window position: {e}")
    
    def destroy(self) -> None:
        """Clean up the overlay window."""
        if self.overlay_window:
            self.overlay_window.destroy()
            self.overlay_window = None


class TranslatorService:
    """Thread-pooled translation service using deep-translator when available."""

    def __init__(self, target_language: str = "ru", max_workers: int = 2) -> None:
        self.target_language = target_language
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="translator")
        self._translation_cache: dict[str, str] = {}

    def translate_async(self, text: str) -> Future:
        return self._pool.submit(self._translate, text)

    def _translate(self, text: str) -> Optional[str]:
        """Translate text using Google Translate with fallback and caching."""
        # Check cache first
        if text in self._translation_cache:
            return self._translation_cache[text]
        
        # Try Google Translate first
        result = translate_text(text, self.target_language)
        if result:
            self._translation_cache[text] = result
            return result
        
        # Fallback to local dictionary
        fallback_result = simple_translate_fallback(text, self.target_language)
        self._translation_cache[text] = fallback_result
        return fallback_result

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)




class CaptionApp:
    """Computer vision-based live captions translator with interactive word detection."""

    def __init__(self, target_language: str = "ru") -> None:
        self.target_language = target_language
        self.root = tk.Tk()
        self.root.title("Live Captions — Computer Vision Translator")
        self.root.geometry("350x170")
        
        # Setup logging
        # Clean up old log files before creating new ones
        cleanup_old_log_files()
        
        self.log_file = f"captions_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_handle = open(self.log_file, "w", encoding="utf-8")
        self.log_handle.write(f"Live Captions Translation Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_handle.write(f"Target Language: {target_language}\n")
        self.log_handle.write("=" * 80 + "\n\n")
        self.log_handle.flush()

        # Simple layout - only translation display
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Translation display area
        self.translation_frame = ttk.Frame(self.root)
        self.translation_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Translation labels
        self.word_translation_var = tk.StringVar(value="Click 'Show' to start word tracking")
        self.word_translation_label = tk.Label(self.translation_frame, textvariable=self.word_translation_var, 
                                             font=("Segoe UI", 14, "bold"), fg="darkgreen", wraplength=350, 
                                             justify="center")
        self.word_translation_label.pack(pady=(40, 10))
        
        # Show Live Caption button (small, in corner)
        self.show_caption_button = tk.Button(
            self.root,
            text="Show",
            font=("Segoe UI", 8),
            bg="lightblue",
            fg="darkblue",
            command=self._on_toggle_caption_clicked,
            width=8,
            height=1
        )
        # Place in top-right corner
        self.show_caption_button.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)
        
        # Reverso button (next to Show button)
        self.reverso_button = tk.Button(
            self.root,
            text="Reverso",
            font=("Segoe UI", 8),
            bg="lightgreen",
            fg="darkgreen",
            command=self._on_reverso_clicked,
            width=8,
            height=1,
            state="disabled"  # Disabled until a word is translated
        )
        # Place next to Show button
        self.reverso_button.place(relx=1.0, rely=0.0, anchor="ne", x=-90, y=10)
        
        # Cambridge button (next to Reverso button)
        self.cambridge_button = tk.Button(
            self.root,
            text="Cambridge",
            font=("Segoe UI", 8),
            bg="lightcoral",
            fg="darkred",
            command=self._on_cambridge_clicked,
            width=8,
            height=1,
            state="disabled"  # Disabled until a word is translated
        )
        # Place next to Reverso button
        self.cambridge_button.place(relx=1.0, rely=0.0, anchor="ne", x=-170, y=10)
        
        self.status_value = tk.StringVar(value="Ready - Click 'Show Live Caption' to start")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_value, anchor="w")
        self.status_bar.grid(row=1, column=0, sticky="ew")

        # Services
        self.translator = TranslatorService(target_language=self.target_language)
        self.word_detector: Optional[WordDetector] = None
        self.live_captions_window: Optional[gw.Win32Window] = None
        self.window_manager: Optional[WindowManager] = None
        
        # Computer vision thread
        self.cv_thread = None
        self.running = False  # Don't start automatically
        self.cv_initialized = False  # Track if CV system is initialized
        
        # Window positioning tracking
        self.last_translation_window_pos = None
        self.positioning_enabled = False  # Don't start positioning automatically

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Track translated words for logging
        self._translated_words: dict[str, str] = {}
        self._word_translations: List[tuple[str, str, str]] = []
        self._words_log_file = f"translated_words_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Current words for Reverso
        self.current_original_word: Optional[str] = None
        self.current_translated_word: Optional[str] = None
        
    def _on_toggle_caption_clicked(self) -> None:
        """Handle the toggle button click - show/hide Live Captions and word detection."""
        if not self.cv_initialized:
            # First try to launch LiveCaptions.exe
            self.status_value.set("Launching LiveCaptions.exe...")
            self.show_caption_button.config(state="disabled", text="Launching...", bg="orange")
            self.root.update()
            
            if launch_live_captions_exe():
                # Wait a bit more for the window to be fully ready
                time.sleep(2)
                # Initialize the computer vision system
                self._initialize_cv_system()
            else:
                # Fallback to the original method if LiveCaptions.exe launch fails
                self.status_value.set("LiveCaptions.exe not found, using Windows Live Captions...")
                self.root.update()
                time.sleep(1)
                self._initialize_cv_system()
        else:
            # Toggle the system on/off
            self._toggle_cv_system()
    
    def _on_reverso_clicked(self) -> None:
        """Handle the Reverso button click - open Reverso with the original English word."""
        if self.current_original_word:
            # URL encode the word to handle special characters
            import urllib.parse
            encoded_word = urllib.parse.quote(self.current_original_word)
            url = f"https://www.reverso.net/text-translation#sl=eng&tl=rus&text={encoded_word}"
            webbrowser.open(url)
            print(f"Opening Reverso for English word: {self.current_original_word}")
        else:
            print("No original word available for Reverso")
    
    def _on_cambridge_clicked(self) -> None:
        """Handle the Cambridge button click - open Cambridge Dictionary with the original English word."""
        if self.current_original_word:
            # URL encode the word to handle special characters
            import urllib.parse
            encoded_word = urllib.parse.quote(self.current_original_word)
            url = f"https://dictionary.cambridge.org/dictionary/english/{encoded_word}"
            webbrowser.open(url)
            print(f"Opening Cambridge Dictionary for English word: {self.current_original_word}")
        else:
            print("No original word available for Cambridge Dictionary")

    def _initialize_cv_system(self) -> None:
        """Initialize the computer vision system and Live Captions window."""
        self.status_value.set("Initializing Live Captions...")
        self.show_caption_button.config(state="disabled", text="Starting...", bg="orange")
        self.root.update()
        
        try:
            # Ensure Live Captions window is visible
            send_hotkey_open_live_captions()
            time.sleep(3)
            
            # Find the Live Captions window
            self.live_captions_window = find_live_captions_window()
            if not self.live_captions_window:
                raise RuntimeError("Live Captions window not found. Please enable Live Captions first.")
            
            # Create window manager
            self.window_manager = WindowManager(self.live_captions_window)
            
            # Position Live Captions window below translation window
            self._position_live_captions_window()
            
            # Create word detector
            self.word_detector = WordDetector(self.live_captions_window, self.window_manager)
            self.word_detector.on_word_selected = self._on_word_selected
            
            # Create overlay window
            self.word_detector.create_overlay_window()
            
            # Start window position monitoring
            self._start_window_position_monitoring()
            
            # Update state BEFORE starting thread
            self.cv_initialized = True
            self.running = True
            self.positioning_enabled = True
            self.word_detector.is_active = True
            
            # Start computer vision thread
            print("Starting computer vision thread...")
            self.cv_thread = threading.Thread(target=self._computer_vision_loop, daemon=True)
            self.cv_thread.start()
            print("Computer vision thread started")
            
            # Update UI
            self.status_value.set("Live Captions active - Click on words to translate")
            self.word_translation_var.set("Click on words in the Live Captions window to see translations")
            self.show_caption_button.config(
                state="normal", 
                text="Hide", 
                bg="lightcoral",
                fg="darkred"
            )
            
        except Exception as e:
            self.status_value.set(f"Error: {e}")
            self.show_caption_button.config(
                state="normal", 
                text="Show",
                bg="lightblue",
                fg="darkblue"
            )
            print(f"Error initializing Live Captions: {e}")

    def _toggle_cv_system(self) -> None:
        """Toggle the computer vision system on/off."""
        if self.word_detector and self.word_detector.is_active:
            # Turn off
            self.word_detector.is_active = False
            self.running = False
            self.positioning_enabled = False
            
            # Destroy overlay window
            if self.word_detector:
                self.word_detector.destroy()
            
            # Hide Live Captions window
            if self.window_manager:
                current_x, current_y = self.window_manager.get_window_position()
                self.window_manager.move_window(current_x, -2000)
            
            # Update UI
            self.status_value.set("Live Captions hidden - Click 'Show' to start")
            self.word_translation_var.set("Click 'Show' to start word tracking")
            self.show_caption_button.config(
                text="Show",
                bg="lightblue",
                fg="darkblue"
            )
        else:
            # Turn on
            self.running = True
            self.positioning_enabled = True
            
            # Show Live Captions window FIRST
            if self.window_manager:
                self._position_live_captions_window()
            
            # Then recreate overlay window at correct position
            if self.word_detector:
                self.word_detector.is_active = True
                # Recreate overlay window after positioning
                self.word_detector.create_overlay_window()
            
            # Check if computer vision thread is still alive, restart if needed
            if not hasattr(self, 'cv_thread') or not self.cv_thread.is_alive():
                print("Restarting computer vision thread...")
                self.cv_thread = threading.Thread(target=self._computer_vision_loop, daemon=True)
                self.cv_thread.start()
                print("Computer vision thread restarted")
            
            # Update UI
            self.status_value.set("Live Captions active - Click on words to translate")
            self.word_translation_var.set("Click on words in the Live Captions window to see translations")
            self.show_caption_button.config(
                text="Hide",
                bg="lightcoral",
                fg="darkred"
            )


    def _computer_vision_loop(self) -> None:
        """Main computer vision loop for word detection."""
        print("CV Loop - Starting computer vision loop")
        while self.running:
            print(f"CV Loop - Running: {self.running}, Active: {self.word_detector.is_active if self.word_detector else 'No detector'}")
            try:
                if not self.live_captions_window or not self.word_detector or not self.window_manager:
                    print("CV Loop - Missing components, waiting...")
                    time.sleep(1)
                    continue
                
                # Only run if word detection is active
                if not self.word_detector.is_active:
                    print("CV Loop - Word detection not active, waiting...")
                    time.sleep(0.5)
                    continue
                
                # Get current Live Captions window position and size
                current_x, current_y = self.window_manager.get_window_position()
                current_width, current_height = self.window_manager.get_window_size()
                
                print(f"CV Loop - Capturing at: ({current_x}, {current_y}) size: {current_width}x{current_height}")
                
                # Update overlay window position to match Live Captions window
                self.word_detector.update_window_position()
                
                # Ensure coordinates are valid for screenshot capture
                if current_width <= 0 or current_height <= 0:
                    print(f"CV Loop - Invalid dimensions: w={current_width}, h={current_height}")
                    time.sleep(0.5)
                    continue
                
                # Check if coordinates are within screen bounds
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                if current_x < 0 or current_y < 0 or current_x >= screen_width or current_y >= screen_height:
                    print(f"CV Loop - Coordinates outside screen bounds: x={current_x}, y={current_y}")
                    time.sleep(0.5)
                    continue
                
                # Capture screenshot of Live Captions window using current coordinates
                try:
                    screenshot = pyautogui.screenshot(region=(
                        current_x,
                        current_y,
                        current_width,
                        current_height
                    ))
                    print(f"CV Loop - Screenshot captured successfully")
                except Exception as e:
                    print(f"CV Loop - Screenshot capture failed: {e}")
                    time.sleep(0.5)
                    continue
                
                # Convert PIL to OpenCV format
                screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                # Detect words
                print("CV Loop - Calling detect_words...")
                print(f"CV Loop - Image shape: {screenshot_cv.shape}")
                word_boxes = self.word_detector.detect_words(screenshot_cv)
                print(f"CV Loop - Detected {len(word_boxes)} words")
                
                if word_boxes:
                    # Draw bounding boxes (with error handling)
                    try:
                        self.word_detector.draw_bounding_boxes()
                        print("CV Loop - Bounding boxes drawn")
                    except Exception as e:
                        print(f"CV Loop - Error drawing bounding boxes: {e}")
                        # Skip this iteration if drawing fails
                        continue
                    for i, word in enumerate(word_boxes[:3]):  # Show first 3 words
                        print(f"  Word {i+1}: '{word['text']}' at {word['bbox']}")
                else:
                    print("CV Loop - No words detected, trying OCR on full image...")
                    # Try to get any text from the image
                    try:
                        import pytesseract
                        text = pytesseract.image_to_string(screenshot_cv, lang="eng").strip()
                        if text:
                            print(f"CV Loop - OCR found text: '{text[:100]}...'")
                        else:
                            print("CV Loop - OCR found no text")
                    except Exception as e:
                        print(f"CV Loop - OCR error: {e}")
                
                time.sleep(0.5)  # Update every 500ms
                
            except Exception as e:
                print(f"Error in computer vision loop: {e}")
                time.sleep(1)
        
    def _on_word_selected(self, word: str) -> None:
        """Handle word selection and trigger translation."""
        print(f"Word selected for translation: '{word}'")
        if not word:
            return
            
        # Update UI
        self.word_translation_var.set(f"{word} → translating...")
        print(f"UI updated with: {word} → translating...")
        
        # Start translation in background
        word_future = self.translator.translate_async(word)
        print(f"Translation started for: '{word}'")

        def on_translation_done(fut: Future) -> None:
            try:
                result = fut.result(timeout=10)
                if result and result.strip():
                    translation = result.strip()
                else:
                    # Fallback translation
                    translation = simple_translate_fallback(word, self.target_language)
                
                # Store both original and translated words for Reverso
                self.current_original_word = word
                self.current_translated_word = translation
                
                # Update UI
                display_text = f"{word} → {translation}"
                self.root.after(0, lambda: self.word_translation_var.set(display_text))
                
                # Enable Reverso and Cambridge buttons
                self.root.after(0, lambda: self.reverso_button.config(state="normal"))
                self.root.after(0, lambda: self.cambridge_button.config(state="normal"))
                
                # Log translation
                timestamp = time.strftime('%H:%M:%S')
                self.log_handle.write(f"[{timestamp}] WORD: {word} → {translation}\n")
                self._translated_words[word.lower()] = translation
                self._word_translations.append((timestamp, word, translation))
                self.log_handle.flush()
                
                print(f"Translation successful: '{word}' -> '{translation}'")
                 
            except Exception as e:
                print(f"Translation error: {e}")
                fallback = simple_translate_fallback(word, self.target_language)
                display_text = f"{word} → {fallback}"
                self.root.after(0, lambda: self.word_translation_var.set(display_text))
                
                # Store both original and fallback translation for Reverso
                self.current_original_word = word
                self.current_translated_word = fallback
                self.root.after(0, lambda: self.reverso_button.config(state="normal"))
                self.root.after(0, lambda: self.cambridge_button.config(state="normal"))
        
        # Run translation in background thread
        threading.Thread(target=on_translation_done, args=(word_future,), daemon=True).start()

    def _position_live_captions_window(self) -> None:
        """Position the Live Captions window directly below the translation window."""
        try:
            if not self.window_manager:
                return
            
            # Get translation window position and size
            translation_x = self.root.winfo_x()
            translation_y = self.root.winfo_y()
            translation_width = self.root.winfo_width()
            translation_height = self.root.winfo_height()
            
            # Calculate position for Live Captions window (directly below)
            live_captions_x = translation_x
            live_captions_y = translation_y + translation_height + 10  # 10px gap
            
            # Move the Live Captions window
            if self.window_manager.move_window(live_captions_x, live_captions_y):
                print(f"Live Captions window positioned at ({live_captions_x}, {live_captions_y})")
                self.last_translation_window_pos = (translation_x, translation_y, translation_width, translation_height)
            else:
                print("Failed to position Live Captions window")
                
        except Exception as e:
            print(f"Error positioning Live Captions window: {e}")

    def _start_window_position_monitoring(self) -> None:
        """Start monitoring the translation window position for automatic repositioning."""
        def monitor_position():
            while self.running and self.positioning_enabled:
                try:
                    # Get current translation window position
                    current_x = self.root.winfo_x()
                    current_y = self.root.winfo_y()
                    current_width = self.root.winfo_width()
                    current_height = self.root.winfo_height()
                    
                    # Check if position has changed
                    if self.last_translation_window_pos is None or \
                       self.last_translation_window_pos != (current_x, current_y, current_width, current_height):
                        
                        # Update Live Captions window position
                        if self.window_manager and self.live_captions_window:
                            live_captions_x = current_x
                            live_captions_y = current_y + current_height + 10  # 10px gap
                            
                            if self.window_manager.move_window(live_captions_x, live_captions_y):
                                print(f"Live Captions window repositioned to ({live_captions_x}, {live_captions_y})")
                                self.last_translation_window_pos = (current_x, current_y, current_width, current_height)
                                
                                # Update overlay window position
                                if self.word_detector:
                                    self.word_detector.update_window_position()
                    
                    time.sleep(0.1)  # Check every 100ms
                    
                except Exception as e:
                    print(f"Error in position monitoring: {e}")
                    time.sleep(1)
        
        # Start monitoring thread
        threading.Thread(target=monitor_position, daemon=True).start()



    def _on_close(self) -> None:
        """Clean up resources when closing the application."""
        try:
            self.running = False
            self.positioning_enabled = False
            self.cv_initialized = False
        except Exception:
            pass
        try:
            if self.word_detector:
                self.word_detector.destroy()
        except Exception:
            pass
        try:
            self.translator.shutdown()
        except Exception:
            pass
        try:
            # Close log file
            self.log_handle.write(f"\nSession ended at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_handle.close()
            print(f"Translation log saved to: {self.log_file}")
            
            # Write summary of translated words to the words log file
            if self._word_translations:
                with open(self._words_log_file, "w", encoding="utf-8") as f:
                    f.write(f"Session ended at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n")
                    f.write("TRANSLATED WORDS SUMMARY:\n")
                    f.write("=" * 80 + "\n")
                    
                    for timestamp, word, translation in self._word_translations:
                        f.write(f"[{timestamp}] {word} → {translation}\n")
                    
                    f.write("=" * 80 + "\n")
                print(f"Translated words log saved to: {self._words_log_file}")
        except Exception as e:
            print(f"Error closing log files: {e}")
        self.root.destroy()

    def run(self) -> None:
        """Start the application main loop."""
        self.root.mainloop()


def run_gui(target_language: str = "ru") -> None:
    app = CaptionApp(target_language=target_language)
    app.run()


if __name__ == "__main__":
    # Launch the GUI by default. The CLI function remains available for debugging.
    try:
        run_gui(target_language="ru")
    except KeyboardInterrupt:
        pass
    
    
    