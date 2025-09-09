# Live Captions Interactive Translator

A real-time caption translation application that captures text from Windows Live Captions and provides interactive word-by-word translation with intelligent sentence boundary detection.

## Features

- **Real-time Caption Capture**: Automatically captures text from Windows Live Captions
- **Interactive Translation**: Click any word to see its translation and the complete sentence
- **Smart Sentence Boundaries**: Uses personal pronouns (we, they, I, he, she, it) to create meaningful translation segments
- **Sliding Window Display**: Shows only the latest 40 words to prevent window overflow
- **Intelligent Auto-scroll**: Automatically scrolls to show new content, pauses when user scrolls up
- **Word Wrapping**: Automatically wraps long sentences to fit within the window
- **Translation Caching**: Caches translations to avoid redundant API calls
- **Comprehensive Logging**: Logs all captions and translations with timestamps

## Requirements

- Windows 10/11 with Live Captions feature
- Python 3.7+
- Tesseract OCR
- Internet connection for Google Translate

## Installation

1. Install Tesseract OCR:
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to default location: `C:\Program Files\Tesseract-OCR\`

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python Caption_live_translater.py
   ```

## Usage

1. **Start Live Captions**: The application will automatically open Windows Live Captions
2. **View Captions**: Captured text appears in the main window with clickable words
3. **Translate Words**: Click any word to see its individual translation and sentence context
4. **Scroll Control**: 
   - Auto-scroll follows new content by default
   - Scroll up to read older captions (auto-scroll pauses)
   - Scroll back to bottom to resume auto-scroll

## Configuration

Edit the configuration section in `Caption_live_translater.py`:

```python
# --- Configuration ---
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SILENCE_SEC: float = 0.3  # Pause time before capturing text
CAPTURE_INTERVAL_SEC: float = 0.2  # How often to check for new text
MAX_WORDS_DISPLAY: int = 40  # Maximum words to display
TEST_MODE = True  # Set to False for production use
```

## File Structure

```
Caption_translater/
├── Caption_live_translater.py    # Main application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── captions_log_*.txt           # Caption logs (auto-generated)
└── translated_words_*.txt       # Translation logs (auto-generated)
```

## Logging

The application creates two types of log files:

- **Caption Logs**: `captions_log_YYYYMMDD_HHMMSS.txt` - Complete session logs
- **Translation Logs**: `translated_words_YYYYMMDD_HHMMSS.txt` - Word translation summaries

## Technical Details

### Caption Processing
- Uses OCR to capture text from Live Captions window
- Implements stability detection to avoid capturing partial text
- Cleans and normalizes text to remove duplicates and artifacts

### Translation System
- Primary: Google Translate API via deep-translator
- Fallback: Local word mapping dictionary
- Caching system to improve performance

### Smart Boundaries
- Splits text at personal pronouns for complete thought segments
- Ensures translations are meaningful and contextually complete
- Prevents fragmented or partial translations

### UI Features
- 40-word sliding window to prevent overflow
- Automatic word wrapping for long sentences
- Smart auto-scroll that respects user interaction
- Real-time translation display

## Troubleshooting

### Common Issues

1. **"Live Captions window not found"**
   - Ensure Windows Live Captions is enabled
   - Try manually opening Live Captions first

2. **"Tesseract not found"**
   - Verify Tesseract is installed at the correct path
   - Update `TESSERACT_EXE` path if needed

3. **No captions appearing**
   - Check if Live Captions is showing text
   - Verify audio is playing and being captioned
   - Try adjusting `SILENCE_SEC` and `CAPTURE_INTERVAL_SEC`

4. **Translation not working**
   - Check internet connection
   - Verify Google Translate API is accessible
   - Check console for error messages

## Development

### Adding New Languages
1. Update the `word_map` in `simple_translate_fallback()`
2. Modify `target_language` parameter
3. Test with the new language

### Customizing Translation
- Modify `translate_text()` for different translation services
- Update `simple_translate_fallback()` for local translations
- Adjust `_split_at_pronoun_boundaries()` for different boundary rules

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Version History

- v1.0: Initial release with basic caption capture and translation
- v1.1: Added smart sentence boundaries and pronoun detection
- v1.2: Implemented sliding window and auto-scroll features
- v1.3: Enhanced word wrapping and user interaction handling
