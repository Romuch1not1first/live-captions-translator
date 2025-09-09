from __future__ import annotations
import ctypes
import time
import re
from typing import Generator, Iterable, Optional, List
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, Future
import tkinter as tk
from tkinter import ttk

import pyautogui
import pygetwindow as gw
import pytesseract

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


class ScrollableFrame(ttk.Frame):
    """A simple vertically scrollable frame for accumulating sentence rows."""

    def __init__(self, master: tk.Widget, *args, **kwargs) -> None:
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self._inner_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Resize inner width when the canvas width changes
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self._inner_window, width=e.width),
        )


class CaptionApp:
    """Tkinter application displaying live captions with per-word translation on click."""

    def __init__(self, target_language: str = "ru") -> None:
        self.target_language = target_language
        self.root = tk.Tk()
        self.root.title("Live Captions — Interactive Translator")
        self.root.geometry("800x250")
        
        # Setup logging
        self.log_file = f"captions_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_handle = open(self.log_file, "w", encoding="utf-8")
        self.log_handle.write(f"Live Captions Translation Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_handle.write(f"Target Language: {target_language}\n")
        self.log_handle.write("=" * 80 + "\n\n")
        self.log_handle.flush()

        # Single-column layout like Live Captions: sentences stack; each row shows
        # translation ABOVE the original caption; window is freely resizable.
        self.root.grid_rowconfigure(1, weight=1)  # Only captions frame should expand
        self.root.grid_columnconfigure(0, weight=1)

        # Translation display area above captions
        self.translation_frame = ttk.Frame(self.root)
        self.translation_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        
        # Translation labels
        self.word_translation_var = tk.StringVar(value="Click a word to see translation")
        self.word_translation_label = tk.Label(self.translation_frame, textvariable=self.word_translation_var, 
                                             font=("Segoe UI", 10), fg="blue", wraplength=800)
        self.word_translation_label.pack(anchor="w", pady=(0, 2))
        
        self.sentence_translation_var = tk.StringVar(value="")
        self.sentence_translation_label = tk.Label(self.translation_frame, textvariable=self.sentence_translation_var, 
                                                 font=("Segoe UI", 12, "bold"), fg="darkgreen", wraplength=800)
        self.sentence_translation_label.pack(anchor="w", pady=(0, 4))

        self.captions_frame = ScrollableFrame(self.root)
        self.captions_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self.status_value = tk.StringVar(value="Starting capture…")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_value, anchor="w")
        self.status_bar.grid(row=2, column=0, sticky="ew")

        # Services
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.capture_thread = CaptionCaptureThread(self.queue)
        self.translator = TranslatorService(target_language=self.target_language)

        self._start_capture()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Track labels to update wraplength on resize
        self._all_wrap_labels: List[tk.Label] = [self.word_translation_label, self.sentence_translation_label]

        # Poll queue for new sentences without blocking UI
        self._poll_queue()

        # Adjust wraplengths dynamically when window is resized
        self.root.bind("<Configure>", self._on_resize)
        
        # Bind scroll events to detect user scrolling
        self.captions_frame.canvas.bind("<MouseWheel>", self._on_user_scroll)
        self.captions_frame.canvas.bind("<Button-4>", self._on_user_scroll)  # Linux scroll up
        self.captions_frame.canvas.bind("<Button-5>", self._on_user_scroll)  # Linux scroll down
        self.captions_frame.canvas.bind("<Button-1>", self._on_user_scroll)  # Click and drag
        self.captions_frame.canvas.bind("<B1-Motion>", self._on_user_scroll)  # Drag motion

        # Keep a small window of recently shown sentences to avoid duplicates
        self._recent_sentences: List[str] = []
        self._recent_canon: List[str] = []
        self._recent_limit: int = 16
        
        # Track translated words for separate logging
        self._translated_words: dict[str, str] = {}  # word -> translation
        self._word_translations: List[tuple[str, str, str]] = []  # (timestamp, word, translation)
        self._words_log_file = f"translated_words_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Sliding window for caption words
        self._word_buffer: List[str] = []  # Buffer to maintain latest words
        self._current_display_row = None  # Reference to current display row
        
        # Smart scrolling state
        self._auto_scroll_enabled = True  # Whether auto-scroll is enabled
        self._last_scroll_position = 1.0  # Last known scroll position (1.0 = bottom)
        self._user_scrolled = False  # Whether user has manually scrolled
        
        # Remove previous logs except translated words
        self._cleanup_old_logs()

    def _cleanup_old_logs(self) -> None:
        """Remove previous log files except translated words logs."""
        try:
            import glob
            import os
            # Find all caption log files in the current directory
            caption_logs = glob.glob("captions_log_*.txt")
            print(f"DEBUG: Found {len(caption_logs)} old caption log files to remove")
            for log_file in caption_logs:
                try:
                    os.remove(log_file)
                    print(f"DEBUG: Removed old caption log: {log_file}")
                except Exception as e:
                    print(f"DEBUG: Could not remove {log_file}: {e}")
            if not caption_logs:
                print("DEBUG: No old caption log files found")
        except Exception as e:
            print(f"DEBUG: Error cleaning up old logs: {e}")

    def _write_word_to_file(self, word: str, translation: str) -> None:
        """Write translated word to the separate words log file."""
        # Don't write individual translations to file - only track them for summary
        # The summary will be written at the end of the session
        pass

    def _start_capture(self) -> None:
        self.capture_thread.start()
        self.status_value.set("Capturing from Live Captions…")
        print("DEBUG: Started capture thread")

    def _poll_queue(self) -> None:
        try:
            while True:
                sentence = self.queue.get_nowait()
                print(f"DEBUG: Received sentence from queue: '{sentence}'")
                if sentence.startswith("__ERROR__:"):
                    self.status_value.set(sentence.replace("__ERROR__:", "Error: "))
                    continue
                
                # Skip duplicate or near-duplicate sentences recently displayed
                normalized = normalize_text(sentence)
                canon = canonicalize_text(sentence)
                
                # Check if this sentence is a subset of a recently displayed sentence
                is_subset = False
                for recent in self._recent_sentences:
                    if sentence.lower() in recent.lower() and len(sentence) < len(recent):
                        print(f"DEBUG: Skipping subset sentence: '{sentence}' (subset of '{recent}')")
                        is_subset = True
                        break
                
                if is_subset:
                    continue
                
                if normalized in self._recent_sentences:
                    print(f"DEBUG: Skipping duplicate normalized sentence: '{normalized}'")
                    continue
                if canon in self._recent_canon:
                    print(f"DEBUG: Skipping duplicate canonical sentence: '{canon}'")
                    continue
                if self._is_near_duplicate(canon):
                    print(f"DEBUG: Skipping near-duplicate sentence: '{canon}'")
                    continue
                    
                self._recent_sentences.append(normalized)
                self._recent_canon.append(canon)
                if len(self._recent_sentences) > self._recent_limit:
                    self._recent_sentences.pop(0)
                if len(self._recent_canon) > self._recent_limit:
                    self._recent_canon.pop(0)
                
                try:
                    print(f"DEBUG: Adding sentence row: '{sentence}'")
                    self._add_sentence_row(sentence)
                except Exception as e:
                    print(f"Error adding sentence row: {e}")
                    # Continue processing other sentences
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in poll_queue: {e}")
            # Continue polling even if there's an error
        finally:
            self.root.after(150, self._poll_queue)

    def _add_sentence_row(self, sentence: str) -> None:
        # Clean the sentence before processing
        cleaned_sentence = self._clean_sentence_for_translation(sentence)
        
        # Log the caption
        timestamp = time.strftime('%H:%M:%S')
        self.log_handle.write(f"[{timestamp}] CAPTION: {cleaned_sentence}\n")
        self.log_handle.flush()
        
        # Split into segments based on personal pronoun boundaries
        segments = self._split_at_pronoun_boundaries(cleaned_sentence)
        
        # Process each segment
        for segment in segments:
            if self._should_translate_segment(segment):
                # Add words from this complete segment to the buffer
                segment_words = self._tokenize_words(segment)
                self._word_buffer.extend(segment_words)
                
                # Log the complete segment
                self.log_handle.write(f"[{timestamp}] SEGMENT: {segment}\n")
                self.log_handle.flush()
        
        # Maintain sliding window of MAX_WORDS_DISPLAY words
        if len(self._word_buffer) > MAX_WORDS_DISPLAY:
            # Remove excess words from the beginning
            excess = len(self._word_buffer) - MAX_WORDS_DISPLAY
            self._word_buffer = self._word_buffer[excess:]
            print(f"DEBUG: Removed {excess} words from beginning, buffer now has {len(self._word_buffer)} words")
        
        # Update or create the display row
        self._update_display_row()

    def _update_display_row(self) -> None:
        """Update the display row with the current word buffer."""
        # Remove existing display row if it exists
        if self._current_display_row:
            try:
                self._current_display_row.destroy()
            except Exception as e:
                print(f"DEBUG: Error destroying display row: {e}")
        
        if not self._word_buffer:
            return
            
        # Create new display row
        row = ttk.Frame(self.captions_frame.inner)
        row.pack(fill="x", pady=3)

        # Word buttons with wrapping - this IS the caption display
        words_frame = tk.Frame(row)
        words_frame.pack(fill="x", pady=0)

        # Create a wrapping layout for words
        self._pack_words_with_wrapping(words_frame, self._word_buffer, " ".join(self._word_buffer))

        # Store reference to this row
        self._current_display_row = row

        # Auto-scroll to the bottom if enabled
        if self._auto_scroll_enabled:
            self._scroll_to_bottom()
        else:
            # Check if user is back at bottom and resume auto-scroll
            self._check_scroll_position()

    def _on_user_scroll(self, event) -> None:
        """Handle user scroll events to manage auto-scroll state."""
        try:
            # Get current scroll position
            canvas = self.captions_frame.canvas
            y_top, y_bot = canvas.yview()
            current_position = y_bot
            
            # Check if user scrolled up from the bottom
            if current_position < 0.95:  # Not at bottom (with small tolerance)
                if not self._user_scrolled:
                    print("DEBUG: User scrolled up, pausing auto-scroll")
                    self._user_scrolled = True
                    self._auto_scroll_enabled = False
            else:  # User is at or near the bottom
                if self._user_scrolled:
                    print("DEBUG: User scrolled back to bottom, resuming auto-scroll")
                    self._user_scrolled = False
                    self._auto_scroll_enabled = True
            
            self._last_scroll_position = current_position
        except Exception as e:
            print(f"DEBUG: Error in scroll detection: {e}")

    def _check_scroll_position(self) -> None:
        """Check if user is at bottom and resume auto-scroll if so."""
        try:
            canvas = self.captions_frame.canvas
            y_top, y_bot = canvas.yview()
            current_position = y_bot
            
            # If user is at bottom, resume auto-scroll
            if current_position >= 0.95:  # At or near bottom
                if self._user_scrolled:
                    print("DEBUG: User at bottom, resuming auto-scroll")
                    self._user_scrolled = False
                    self._auto_scroll_enabled = True
                    self._scroll_to_bottom()
        except Exception as e:
            print(f"DEBUG: Error checking scroll position: {e}")

    def _tokenize_words(self, sentence: str) -> List[str]:
        # Keep apostrophes within words (e.g., don't, I'm). Remove surrounding punctuation.
        tokens = re.findall(r"[A-Za-z0-9']+", sentence)
        return [t for t in tokens if t.strip()]

    def _split_at_pronoun_boundaries(self, text: str) -> List[str]:
        """Split text at personal pronoun boundaries to create complete thought segments."""
        # Personal pronouns that mark natural sentence boundaries
        pronouns = ['we', 'they', 'i', 'he', 'she', 'it']
        
        # Split text into words
        words = self._tokenize_words(text)
        if not words:
            return []
        
        segments = []
        current_segment = []
        
        for word in words:
            current_segment.append(word)
            
            # Check if this word is a personal pronoun (case insensitive)
            if word.lower() in pronouns:
                # Complete the current segment
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []
        
        # Add any remaining words as the final segment
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments

    def _should_translate_segment(self, segment: str) -> bool:
        """Determine if a segment should be translated based on its completeness."""
        if not segment or len(segment.strip()) < 3:  # Too short
            return False
        
        # Check if segment ends with a personal pronoun
        words = self._tokenize_words(segment)
        if not words:
            return False
            
        pronouns = ['we', 'they', 'i', 'he', 'she', 'it']
        last_word = words[-1].lower()
        
        # Translate if it ends with a pronoun or is a complete thought
        return last_word in pronouns or len(words) >= 5

    def _pack_words_with_wrapping(self, parent_frame: tk.Frame, words: List[str], sentence: str) -> None:
        """Pack words with automatic wrapping to fit within the window width."""
        if not words:
            return
            
        # Get the available width from the main window
        try:
            # Get the width of the captions frame (which is the scrollable area)
            main_width = self.captions_frame.winfo_width()
            if main_width <= 0:
                main_width = self.root.winfo_width()
            available_width = main_width - 40  # 40px padding for scrollbar and margins
            if available_width <= 0:
                available_width = 600  # Default width if not available
        except:
            available_width = 600  # Default width
        
        # Font for measuring
        font = ("Segoe UI", 15, "normal")
        
        current_line = tk.Frame(parent_frame)
        current_line.pack(fill="x", pady=1)
        current_width = 0
        
        for word in words:
            # Create a temporary button to measure its width
            temp_btn = tk.Button(
                current_line,
                text=word,
                font=font,
                padx=2,
                pady=2
            )
            temp_btn.update_idletasks()
            word_width = temp_btn.winfo_reqwidth()
            temp_btn.destroy()
            
            # If adding this word would exceed the width, start a new line
            if current_width + word_width > available_width and current_width > 0:
                current_line = tk.Frame(parent_frame)
                current_line.pack(fill="x", pady=1)
                current_width = 0
            
            # Create the actual button
            btn = tk.Button(
                current_line,
                text=word,
                command=lambda w=word, s=sentence: self._on_word_clicked(w, s),
                bd=0,
                highlightthickness=0,
                padx=2,
                pady=2,
                relief="flat",
                bg="lightgray",
                fg="black",
                font=font
            )
            btn.pack(side="left", padx=0, pady=0)
            current_width += word_width

    def _clean_sentence_for_translation(self, sentence: str) -> str:
        """Clean sentence by removing repeated blocks and normalizing for translation."""
        # First, normalize the text
        normalized = normalize_text(sentence)
        
        # Split into words and apply the same cleaning logic as display
        words = self._tokenize_words(normalized)
        words = self._collapse_repeated_blocks(words)
        words = self._collapse_consecutive_duplicates(words)
        
        # Join back into a clean sentence
        cleaned = " ".join(words)
        
        # Additional cleanup: remove very long repeated phrases
        # Look for patterns like "word word word word" and reduce to "word word"
        cleaned = re.sub(r'\b(\w+(?:\s+\w+){1,3})\s+\1\b', r'\1', cleaned)
        
        return cleaned

    def _on_word_clicked(self, word: str, sentence: str) -> None:
        # Clean the sentence to remove repeated blocks before translation
        cleaned_sentence = self._clean_sentence_for_translation(sentence)
        
        # Find the complete segment containing this word
        segments = self._split_at_pronoun_boundaries(cleaned_sentence)
        word_segment = None
        
        for segment in segments:
            if word.lower() in segment.lower():
                word_segment = segment
                break
        
        # If no segment found, use the full sentence
        if not word_segment:
            word_segment = cleaned_sentence
        
        # Update the main translation display area
        self.word_translation_var.set(f"{word} → translating…")
        self.sentence_translation_var.set("Translating segment…")

        word_future = self.translator.translate_async(word)
        sent_future = self.translator.translate_async(word_segment)

        def on_done(fut: Future, setter: tk.StringVar, label_prefix: Optional[str] = None) -> None:
            try:
                result = fut.result(timeout=10)  # Increased timeout for better reliability
                if result and result.strip():
                    text = result.strip()
                else:
                    # Try fallback translation
                    if label_prefix:
                        # Word translation
                        text = simple_translate_fallback(word, self.target_language)
                    else:
                        # Segment translation with highlighted word
                        text = translate_sentence_with_highlighted_word(word_segment, word, self.target_language)
            except Exception as e:
                print(f"Translation error: {e}")
                # Try fallback translation
                if label_prefix:
                    # Word translation
                    text = simple_translate_fallback(word, self.target_language)
                else:
                    # Segment translation with highlighted word
                    text = translate_sentence_with_highlighted_word(word_segment, word, self.target_language)
            
            # Log the translation
            timestamp = time.strftime('%H:%M:%S')
            if label_prefix:
                display = f"{label_prefix} {text}"
                self.log_handle.write(f"[{timestamp}] WORD: {word} → {text}\n")
                # Track translated words for separate file
                self._translated_words[word.lower()] = text
                self._word_translations.append((timestamp, word, text))
                self._write_word_to_file(word, text)
            else:
                display = text
                self.log_handle.write(f"[{timestamp}] SEGMENT: {word_segment} → {text}\n")
            self.log_handle.flush()
            
            self.root.after(0, lambda: setter.set(display))

        # Attach callbacks in background threads; results marshalled back via after()
        threading.Thread(target=on_done, args=(word_future, self.word_translation_var, f"{word} →"), daemon=True).start()
        threading.Thread(target=on_done, args=(sent_future, self.sentence_translation_var, None), daemon=True).start()

    def _is_near_duplicate(self, canon_sentence: str) -> bool:
        """Return True if canon_sentence is very similar to any recent canonical one (Jaccard >= 0.85)."""
        current_words = set(canon_sentence.split())
        if not current_words:
            return False
        for prev in self._recent_canon:
            prev_words = set(prev.split())
            if not prev_words:
                continue
            inter = len(current_words & prev_words)
            union = len(current_words | prev_words)
            if union and (inter / union) >= 0.85:
                return True
        return False

    def _on_resize(self, _event: tk.Event) -> None:
        # Compute available width inside captions frame to wrap labels nicely
        try:
            # Subtract scrollbar width and padding
            avail = max(200, self.captions_frame.winfo_width() - 24)
        except Exception:
            avail = 700
        for lbl in self._all_wrap_labels:
            try:
                lbl.configure(wraplength=avail)
            except Exception:
                pass
        
        # Re-wrap all caption rows when window is resized
        self._rewrap_all_captions()

    def _rewrap_all_captions(self) -> None:
        """Re-wrap all caption rows when window is resized."""
        try:
            # This is a placeholder for now - in a full implementation,
            # we would need to store the original text and re-create the word layouts
            # For now, the wrapping happens when captions are first created
            pass
        except Exception as e:
            print(f"DEBUG: Error re-wrapping captions: {e}")

    def _scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the caption window."""
        try:
            canvas = self.captions_frame.canvas
            
            def do_scroll() -> None:
                try:
                    canvas.yview_moveto(1.0)
                    # Update scroll position tracking
                    self._last_scroll_position = 1.0
                except Exception:
                    pass

            # Ensure layout is updated, then scroll; schedule an extra tick as backup
            self.root.update_idletasks()
            self.root.after_idle(do_scroll)
            self.root.after(60, do_scroll)
        except Exception:
            pass

    def _collapse_consecutive_duplicates(self, words: List[str]) -> List[str]:
        if not words:
            return words
        result: List[str] = [words[0]]
        for w in words[1:]:
            if w != result[-1]:
                result.append(w)
        return result

    def _collapse_repeated_blocks(self, words: List[str]) -> List[str]:
        n = len(words)
        if n < 4:
            return words
        # Try to find the smallest period k such that words == block * m (m>=2)
        for k in range(1, n // 2 + 1):
            if n % k != 0:
                continue
            block = words[:k]
            if block * (n // k) == words:
                return block
        return words

    def _on_close(self) -> None:
        try:
            self.capture_thread.stop()
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
    
    
    