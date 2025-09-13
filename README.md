# Live Captions Computer Vision Translator

A real-time computer vision-based caption translation application that uses OCR to detect and translate individual words from Windows Live Captions with interactive click-to-translate functionality. Features multiple AI translation APIs with secure encrypted storage.

## 🚀 Features

### Core Functionality
- **Computer Vision Word Detection**: Uses OCR to detect individual words in the Live Captions window
- **Interactive Click-to-Translate**: Click on any detected word to see its translation instantly
- **Real-time Bounding Boxes**: Visual bounding boxes around detected words with hover effects
- **Live Captions Integration**: Automatically positions and manages the Live Captions window
- **Hide/Show Functionality**: Toggle Live Captions visibility with a corner button

### AI Translation APIs
- **Perplexity AI**: Primary translation service with context-aware translation
- **Gemini AI**: Secondary translation service with Google's advanced AI
- **Deep Translator**: Fallback service using Google Translate
- **Context-Aware Translation**: Translates words based on full sentence context
- **Smart Fallback Chain**: Automatically switches between APIs if one fails

### Security & Storage
- **🔐 Encrypted API Key Storage**: AES-256 encryption for all sensitive data
- **System-Specific Encryption**: Keys only work on the original system
- **Automatic Migration**: Seamlessly upgrades from old unencrypted files
- **Secure File Format**: All data stored in `.secure` encrypted files

### User Interface
- **Integrated Settings Panel**: Built-in API selection and key management
- **API Selection**: Choose between Perplexity AI and Gemini AI
- **Key Validation**: Test API keys before saving
- **Auto-Close Settings**: Settings panel closes automatically after successful save
- **Clean UI**: Minimalist interface with corner toggle button

### Performance & Reliability
- **Translation Caching**: Caches translations to avoid redundant API calls
- **Asynchronous Processing**: Non-blocking translation requests
- **Comprehensive Logging**: Logs all word translations with timestamps
- **Error Handling**: Robust error handling with fallback mechanisms

## 📋 Requirements

- Windows 10/11 with Live Captions feature
- Python 3.7+
- Tesseract OCR
- Internet connection for AI translation APIs
- API keys for Perplexity AI and/or Gemini AI

## 🛠️ Installation

### 1. Install Tesseract OCR
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install to default location: `C:\Program Files\Tesseract-OCR\`

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Get API Keys
- **Perplexity AI**: Get your API key from https://www.perplexity.ai/settings/api
- **Gemini AI**: Get your API key from https://makersuite.google.com/app/apikey

### 4. Run the Application
```bash
python Caption_live_translater.py
```

## 🎯 Usage

### Basic Usage
1. **Start the Application**: Run the script to open the translation window
2. **Configure APIs**: Click the Settings button to configure your API keys
3. **Select API**: Choose between Perplexity AI or Gemini AI
4. **Test & Save**: Enter your API key and click "Save & Test"
5. **Show Live Captions**: Click the "Show" button in the top-right corner
6. **Translate Words**: Click any detected word to see its translation instantly

### Settings Configuration
1. **Open Settings**: Click the Settings button in the main window
2. **Select API**: Choose your preferred translation API (Perplexity AI or Gemini AI)
3. **Enter API Key**: Paste your API key in the appropriate field
4. **Save & Test**: Click "Save & Test" to validate and save your settings
5. **Auto-Close**: Settings panel closes automatically after successful save

## 🔧 Configuration

### API Configuration
The application supports multiple translation APIs with automatic fallback:

```python
# Primary API (selected in settings)
selected_api = "perplexity"  # or "gemini"

# Fallback chain: Primary API → Secondary API → Deep Translator
```

### OCR Configuration
Edit the configuration section in `Caption_live_translater.py`:

```python
# --- Configuration ---
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SILENCE_SEC: float = 0.3  # Pause time before capturing text
CAPTURE_INTERVAL_SEC: float = 0.2  # How often to check for new text
MAX_WORDS_DISPLAY: int = 40  # Maximum words to display
TEST_MODE = True  # Set to False for production use
```

## 🔐 Security Features

### Encrypted Storage
- **AES-256 Encryption**: All API keys encrypted with industry-standard encryption
- **System-Specific Keys**: Encryption keys derived from system characteristics
- **PBKDF2 Key Derivation**: 100,000 iterations for strong key derivation
- **Secure File Format**: Files stored with `.secure` extension

### Protected Files
- `api_key.secure` - Encrypted Perplexity API key
- `gemini_api_key.secure` - Encrypted Gemini API key
- `api_selection.secure` - Encrypted API selection

### Security Benefits
- **Theft Protection**: API keys are useless if files are stolen
- **System Binding**: Keys only work on the original system
- **Automatic Cleanup**: Removes unencrypted files after migration

## 📁 File Structure

```
Caption_translater/
├── Caption_live_translater.py    # Main application
├── requirements.txt              # Python dependencies
├── setup.py                     # Package setup
├── README.md                    # This file
├── LICENSE                      # License file
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guidelines
├── install.bat                  # Windows installation script
├── api_key.secure               # Encrypted Perplexity API key
├── gemini_api_key.secure        # Encrypted Gemini API key
├── api_selection.secure          # Encrypted API selection
└── .gitignore                   # Git ignore rules
```

## 🧠 How It Works

### Computer Vision Pipeline
1. **Screenshot Capture**: Captures Live Captions window region
2. **OCR Processing**: Uses Tesseract to extract word-level data
3. **Word Detection**: Filters words by confidence and creates bounding boxes
4. **Interactive Overlay**: Creates transparent overlay window for click detection
5. **Context Extraction**: Extracts full sentence context for better translation
6. **AI Translation**: Translates clicked words using selected AI API

### Translation System
1. **Primary API**: Uses selected API (Perplexity AI or Gemini AI)
2. **Context-Aware**: Translates words based on full sentence context
3. **Fallback Chain**: Automatically switches to secondary API if primary fails
4. **Final Fallback**: Uses Deep Translator if both AI APIs fail
5. **Caching**: Stores translations to avoid redundant API calls

### Security System
1. **Key Derivation**: Creates system-specific encryption keys
2. **Encryption**: Encrypts all sensitive data with AES-256
3. **Secure Storage**: Saves encrypted data to `.secure` files
4. **Migration**: Automatically migrates old unencrypted files

## 🔍 Troubleshooting

### Common Issues

1. **"Live Captions window not found"**
   - Ensure Windows Live Captions is enabled
   - Try manually opening Live Captions first

2. **"Tesseract not found"**
   - Verify Tesseract is installed at the correct path
   - Update `TESSERACT_EXE` path if needed

3. **"API key is invalid"**
   - Check your API key is correct
   - Verify the API key has proper permissions
   - Test the API key in the settings panel

4. **"No words detected"**
   - Check if Live Captions is showing text
   - Verify audio is playing and being captioned
   - Try adjusting OCR confidence threshold

5. **Translation not working**
   - Check internet connection
   - Verify API keys are valid
   - Check console for error messages

### API-Specific Issues

**Perplexity AI:**
- Ensure API key has proper permissions
- Check API usage limits
- Verify network connectivity

**Gemini AI:**
- Verify API key is from Google AI Studio
- Check API quotas and limits
- Ensure proper API key format

## 🛠️ Development

### Adding New Translation APIs
1. Create new translation function following the pattern:
   ```python
   def _translate_with_new_api(text: str, target_language: str, sentence: str, api_key: str) -> Optional[str]:
       # Implementation here
   ```
2. Add API selection option in settings
3. Update fallback chain in `translate_text()`

### Customizing Detection
- Modify OCR confidence threshold in `detect_words()`
- Adjust bounding box colors and styles
- Customize word filtering criteria

### UI Customization
- Modify button appearance and positioning
- Adjust overlay window transparency
- Customize translation display format

## 📊 Logging

The application creates comprehensive log files:

- **Caption Logs**: `captions_log_YYYYMMDD_HHMMSS.txt` - Complete session logs
- **Translation Logs**: `translated_words_YYYYMMDD_HHMMSS.txt` - Word translation summaries

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to modify and distribute according to your needs.

## 🗺️ Roadmap

### Completed Features ✅
- [x] Computer vision word detection
- [x] Interactive click-to-translate
- [x] Multiple AI translation APIs
- [x] Context-aware translation
- [x] Encrypted API key storage
- [x] Integrated settings panel
- [x] API key validation
- [x] Automatic fallback system

### Future Enhancements 🚀
- [ ] Support for more translation APIs
- [ ] Custom translation models
- [ ] Batch translation mode
- [ ] Translation history
- [ ] Multi-language support
- [ ] Voice synthesis for translations

## 📈 Version History

- **v3.0**: Added multiple AI APIs (Perplexity AI, Gemini AI) with secure storage
- **v2.3**: Enhanced UI with corner button and clean design
- **v2.2**: Implemented hide/show toggle and window management
- **v2.1**: Added interactive click-to-translate functionality
- **v2.0**: Complete rewrite with computer vision word detection

## 🆘 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are properly installed
4. Verify API keys are valid and have proper permissions

For additional support, please open an issue on GitHub.