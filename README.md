# Live Captions Computer Vision Translator

A real-time computer vision-based caption translation application that uses OCR to detect and translate individual words from Windows Live Captions with interactive click-to-translate functionality.

## Features

- **Computer Vision Word Detection**: Uses OCR to detect individual words in the Live Captions window
- **Interactive Click-to-Translate**: Click on any detected word to see its translation instantly
- **Real-time Bounding Boxes**: Visual bounding boxes around detected words with hover effects
- **Live Captions Integration**: Automatically positions and manages the Live Captions window
- **Hide/Show Functionality**: Toggle Live Captions visibility with a corner button
- **Translation Caching**: Caches translations to avoid redundant API calls
- **Comprehensive Logging**: Logs all word translations with timestamps
- **Clean UI**: Minimalist interface with corner toggle button

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

1. **Start the Application**: Run the script to open the translation window
2. **Show Live Captions**: Click the "Show" button in the top-right corner
3. **View Detected Words**: Bounding boxes will appear around detected words
4. **Translate Words**: Click any word to see its translation instantly
5. **Hide/Show**: Use the corner button to toggle Live Captions visibility

## How It Works

### Computer Vision System
- Captures screenshots of the Live Captions window
- Uses Tesseract OCR to detect individual words
- Draws interactive bounding boxes around detected words
- Tracks mouse clicks on bounding boxes for translation

### Word Detection
- Real-time OCR processing of Live Captions content
- Confidence-based filtering to ensure accurate detection
- Automatic bounding box positioning and sizing
- Hover effects for better user experience

### Translation System
- Primary: Google Translate API via deep-translator
- Fallback: Local word mapping dictionary
- Asynchronous translation to prevent UI blocking
- Translation caching for improved performance

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
├── setup.py                     # Package setup
├── README.md                    # This file
├── LICENSE                      # License file
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guidelines
└── .gitignore                   # Git ignore rules
```

## Logging

The application creates two types of log files:

- **Caption Logs**: `captions_log_YYYYMMDD_HHMMSS.txt` - Complete session logs
- **Translation Logs**: `translated_words_YYYYMMDD_HHMMSS.txt` - Word translation summaries

## Technical Details

### Computer Vision Pipeline
1. **Screenshot Capture**: Captures Live Captions window region
2. **OCR Processing**: Uses Tesseract to extract word-level data
3. **Word Detection**: Filters words by confidence and creates bounding boxes
4. **Interactive Overlay**: Creates transparent overlay window for click detection
5. **Translation**: Translates clicked words using Google Translate API

### Window Management
- Automatic Live Captions window positioning
- Real-time window position tracking
- Overlay window synchronization
- Hide/show functionality with proper cleanup

### UI Components
- Main translation window with corner toggle button
- Semi-transparent overlay window for word interaction
- Real-time translation display
- Status bar with system information

## Troubleshooting

### Common Issues

1. **"Live Captions window not found"**
   - Ensure Windows Live Captions is enabled
   - Try manually opening Live Captions first

2. **"Tesseract not found"**
   - Verify Tesseract is installed at the correct path
   - Update `TESSERACT_EXE` path if needed

3. **No words detected**
   - Check if Live Captions is showing text
   - Verify audio is playing and being captioned
   - Try adjusting OCR confidence threshold

4. **Click detection not working**
   - Ensure overlay window is visible
   - Check if bounding boxes are being drawn
   - Verify mouse click coordinates

5. **Translation not working**
   - Check internet connection
   - Verify Google Translate API is accessible
   - Check console for error messages

## Development

### Adding New Languages
1. Update the `word_map` in `simple_translate_fallback()`
2. Modify `target_language` parameter
3. Test with the new language

### Customizing Detection
- Modify OCR confidence threshold in `detect_words()`
- Adjust bounding box colors and styles
- Customize word filtering criteria

### UI Customization
- Modify button appearance and positioning
- Adjust overlay window transparency
- Customize translation display format

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Version History

- v2.0: Complete rewrite with computer vision word detection
- v2.1: Added interactive click-to-translate functionality
- v2.2: Implemented hide/show toggle and window management
- v2.3: Enhanced UI with corner button and clean design