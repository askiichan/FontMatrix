## **Software Requirements Specification (SRS): Gradio Font Comparator**

**Project:** Gradio Font Comparator
**Version:** 1.0
**Date:** 2025-08-15

### 1. Introduction 📜

#### 1.1. Purpose
This document outlines the requirements for the **Gradio Font Comparator**. The project's goal is to develop an interactive web application using Python and the **Gradio** library. The application will allow users to upload multiple font files and generate a comparison table displaying all supported characters and their corresponding Unicode codes (字元碼).

#### 1.2. Scope
The application will provide a web-based UI where users can upload up to **five font files** (e.g., TTF, OTF). The backend will process these files, identify the complete set of characters supported across all fonts, and display a comparison table. The table will show each character, its Unicode code, and indicate which of the uploaded fonts supports it.

#### 1.3. Target Audience
The intended users are **designers, developers, and typographers** who need to quickly compare the character sets of different fonts to ensure complete language support for a project.

---

### 2. Functional Requirements ⚙️

#### 2.1. User Interface (UI) powered by Gradio
* The application must be built using the **Gradio** library.
* **Input Components:**
    * A `gr.File()` component configured to allow multiple file uploads (`file_count="multiple"`), limited to a maximum of 5 files.
    * The component should accept common font file types (e.g., `.ttf`, `.otf`).
* **Output Components:**
    * A `gr.DataFrame()` or `gr.HTML()` component to display the final comparison table. The table must be sortable and searchable.
* **Control Components:** A "Compare Fonts" button to trigger the analysis and a "Clear" button to reset the interface.

#### 2.2. Core Application Logic
* The core Python function will accept a list of uploaded font file paths from the Gradio interface.
* It will use the **`fontTools`** library to parse each font file and extract the set of supported Unicode codepoints from each font's character map (`cmap`).
* The logic must compute the **union** of all character sets to create a master list of all unique characters across all uploaded fonts.
* It will then construct a data structure (e.g., a Pandas DataFrame) with the following columns:
    * `Character` (the actual character glyph)
    * `Unicode` (the character code, e.g., "U+4E00")
    * One column for each uploaded font (e.g., `Font1.ttf`, `Font2.ttf`, etc.), indicating support with a symbol like "✔" or "❌".

#### 2.3. Input/Output Handling
* **Inputs:** The application must handle up to five uploaded font files simultaneously.
* **Outputs:** The results must be rendered in a clear, tabular format. The table should be able to handle potentially thousands of rows (characters) without crashing.
* **Error Handling:** The UI must display a clear error message if a user uploads a non-font file, an unsupported font format, or if any other processing error occurs.

---

### 3. Non-Functional Requirements 🚀

#### 3.1. Performance
* The application should process 5 typical CJK font files (which can be large) and render the comparison table within **15-20 seconds**.
* The Gradio interface should load in under **3 seconds**.

#### 3.2. Usability
* The interface should be self-explanatory: a clear upload area and a visible output table.
* The output table must be easy to read and navigate. Column headers should clearly identify each font file. 

#### 3.3. Technical Stack
* **Programming Language:** Python 3.9+
* **Core Libraries:** `gradio`, `fontTools`, `pandas`
* **Deployment:** The application will be suitable for deployment on Hugging Face Spaces or a private server.

---

### 4. Project Plan & Milestones 🗓️

#### 4.1. Phase 1: Font Parsing Logic
* Develop a Python script that takes a list of font files and generates a comparison DataFrame using `fontTools` and `pandas`.
* **Deadline:** [Date]

#### 4.2. Phase 2: Gradio UI Integration
* Build the Gradio interface with the file upload and DataFrame/HTML output components.
* Connect the backend font parsing logic to the UI.
* **Deadline:** [Date]

#### 4.3. Phase 3: Testing & Deployment
* Test the application with various font files, including edge cases (corrupted files, large CJK fonts).
* Deploy the final application.
* **Deadline:** [Date]

