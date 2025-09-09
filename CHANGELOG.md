# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-09-09

### Added
- Smart sentence boundary detection using personal pronouns (we, they, I, he, she, it)
- Complete thought segment translation for better context
- Intelligent auto-scroll that respects user interaction
- 40-word sliding window to prevent display overflow
- Comprehensive logging system with separate word translation logs
- Word wrapping for long sentences
- Translation caching to improve performance

### Changed
- Improved caption processing to create meaningful translation segments
- Enhanced word clicking to translate complete thought segments
- Better text cleaning and duplicate removal
- Optimized OCR capture with stability detection

### Fixed
- Resolved duplicate translation issues
- Fixed caption display problems
- Improved text cleaning and normalization
- Better handling of repeated text blocks

## [1.2.0] - 2025-09-09

### Added
- Sliding window display with 40-word limit
- Smart auto-scroll functionality
- Word wrapping for long captions
- Enhanced debug output

### Changed
- Improved caption capture logic
- Better text processing and cleaning
- Enhanced user interaction handling

## [1.1.0] - 2025-09-09

### Added
- Personal pronoun boundary detection
- Complete thought segment processing
- Enhanced translation logic
- Better sentence boundary recognition

### Changed
- Improved caption processing
- Enhanced translation accuracy
- Better text segmentation

## [1.0.0] - 2025-09-09

### Added
- Initial release
- Basic caption capture from Windows Live Captions
- Word-by-word translation functionality
- Google Translate integration
- Fallback translation dictionary
- Real-time caption display
- Interactive word clicking
- Basic logging system

### Features
- OCR text capture from Live Captions window
- Real-time translation display
- Word-level translation on click
- Sentence-level translation context
- Basic text cleaning and normalization
- Simple logging to text files
