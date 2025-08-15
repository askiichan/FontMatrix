# FontMatrix - Font Comparison Tool

A program that comparing character coverage across multiple font files.

## ✨ Features

### Core Functionality
- **Multi-Font Analysis**: Upload up to 5 font files (.ttf, .otf) simultaneously
- **Unicode Coverage Comparison**: Comprehensive character-by-character support matrix
- **Interactive Data Table**: Sortable, filterable comparison table with real-time updates
- **Smart Search**: Filter by character, Unicode code (e.g., `U+4E00`, `4E00`), or text patterns
- **Visual Font Previews**: Live text rendering with each uploaded font
- **Export Capabilities**: Download results as CSV with robust Unicode handling

### Advanced Features
- **Difference Analysis**: Show only characters where fonts differ in support
- **Dynamic Sorting**: Sort by Unicode, character, coverage count, or individual font support
- **Performance Optimized**: Handles large CJK fonts efficiently with configurable row limits
- **Robust Error Handling**: Clear feedback for unsupported files or processing errors
- **Responsive UI**: Clean, modern interface that works across devices

## 🏗️ Architecture

The application follows a modular, object-oriented design:

```
FontMatrix/
├── FontProcessor      # Font file parsing and codepoint extraction
├── UnicodeHelper      # Unicode character processing and formatting  
├── ImageRenderer      # Text rendering and image generation
├── DataFrameBuilder   # Data manipulation and filtering
├── CSVExporter        # Robust CSV export with encoding handling
├── FontMatrixApp      # Main application coordinator
└── UIBuilder          # Gradio interface construction
```


## 🔧 Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows, macOS, or Linux
- **Dependencies**: See `requirements.txt`



## 🚀 Installation & Setup

### Option 1: Quick Start
```bash
# Clone or download the project
cd FontMatrix

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Option 2: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

## 📖 Usage Guide

### Getting Started
1. **Launch**: Run `python app.py` and open the displayed URL (typically `http://127.0.0.1:7860`)
2. **Upload Fonts**: Select up to 5 font files (.ttf or .otf)
3. **Compare**: Click "Compare Fonts" to generate the analysis

### Interface Overview

#### Input Controls
- **Font Upload**: Multi-file selector for .ttf/.otf files
- **Sample Text**: Custom text for font previews (defaults to multilingual sample)
- **Preview Height**: Adjustable slider for preview image size
- **Search/Filter**: Text input for filtering results
- **Show Differences**: Toggle to display only characters with different support
- **Sort Options**: Dropdown for sorting by various criteria
- **Row Limit**: Performance slider for large datasets

#### Output Sections
- **Comparison Table**: Interactive data grid showing character support matrix
- **Summary Statistics**: Processing time, row counts, and font information
- **CSV Download**: Export current view with proper Unicode encoding
- **Font Previews**: Visual rendering of sample text in each font

### Advanced Usage

#### Search & Filter Examples
```
Search for specific characters:
- 一 (direct character input)
- U+4E00 (Unicode format)
- 4E00 (hex without prefix)
- CJK (substring search)
```

#### Performance Tips
- Use row limits (default: 5000) for large font collections
- Enable "Show only differences" to focus on coverage gaps
- Sort by "CoverageCount" to identify most/least supported characters

## 🔍 Use Cases

### Typography & Design
- **Language Support Verification**: Ensure fonts support required character sets
- **Font Pairing**: Compare complementary fonts for consistent coverage
- **Multilingual Projects**: Validate support across different languages

### Development & Localization
- **Web Font Selection**: Choose fonts with appropriate Unicode coverage
- **Application Testing**: Verify character support before deployment
- **Font Subsetting**: Identify characters for custom font builds

### Research & Analysis
- **Font Coverage Studies**: Academic research on typography and Unicode
- **Historical Font Analysis**: Compare legacy and modern font support
- **Character Set Documentation**: Generate comprehensive coverage reports

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

Disclaimer: This tool is intended for educational and development purposes. For official use, ensure compliance with all local regulations.
