import os
import io
import base64
import tempfile
import time
import unicodedata
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import gradio as gr
from fontTools.ttLib import TTFont
from fontTools.ttLib.ttFont import TTLibError
from PIL import Image, ImageDraw, ImageFont


# Constants
CHECK = "✔"
CROSS = "❌"
DEFAULT_SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog — 123 一二三四"
MAX_FONTS = 5
SUPPORTED_EXTENSIONS = {".ttf", ".otf"}


@dataclass
class FontInfo:
    """Contains information about a font file."""
    name: str
    path: str
    codepoints: Set[int]


class FontProcessor:
    """Handles font file processing and codepoint extraction."""
    
    @staticmethod
    def is_valid_font_extension(path: str) -> bool:
        """Check if file has a valid font extension."""
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
    
    @staticmethod
    def extract_codepoints(font_path: str) -> Set[int]:
        """Extract supported Unicode codepoints from a font's cmap tables."""
        codepoints: Set[int] = set()
        
        try:
            with TTFont(font_path, lazy=True) as font:
                cmap = font["cmap"]
                for table in cmap.tables:
                    try:
                        # Prefer unicode cmaps
                        if hasattr(table, "isUnicode"):
                            is_unicode = (table.isUnicode() if callable(table.isUnicode) 
                                        else bool(table.isUnicode))
                        else:
                            # Fallback heuristic
                            is_unicode = table.platformID in (0, 3)
                        
                        if is_unicode and table.cmap:
                            codepoints.update(table.cmap.keys())
                    except Exception:
                        # Be resilient to odd cmap tables
                        continue
        except Exception as e:
            raise ValueError(f"Failed to read font file: {e}")
        
        return codepoints
    
    @staticmethod
    def create_unique_names(paths: List[str]) -> List[str]:
        """Create unique, readable names for each path, deduplicating with (n)."""
        seen = {}
        names = []
        
        for path in paths:
            base = Path(path).name
            if base not in seen:
                seen[base] = 1
                names.append(base)
            else:
                seen[base] += 1
                stem = Path(base).stem
                suffix = Path(base).suffix
                name = f"{stem} ({seen[base]}){suffix}"
                names.append(name)
        
        return names


class UnicodeHelper:
    """Handles Unicode character processing and formatting."""
    
    @staticmethod
    def safe_chr(codepoint: int) -> str:
        """Convert codepoint to character, handling problematic cases."""
        try:
            char = chr(codepoint)
            category = unicodedata.category(char)
            
            # Skip control characters, surrogates, non-characters, etc.
            if category.startswith("C") and category not in {"Cf"}:
                return ""
            
            # Handle control characters that might cause CSV issues
            if ord(char) < 32 and char not in {'\t', '\n', '\r'}:
                return f"\\u{codepoint:04X}"
            
            return char
        except Exception:
            return f"\\u{codepoint:04X}"
    
    @staticmethod
    def format_unicode(codepoint: int) -> str:
        """Format codepoint as U+XXXX."""
        return f"U+{codepoint:04X}"


class ImageRenderer:
    """Handles text rendering and image generation."""
    
    @staticmethod
    def render_text_image(text: str, font_path: str, font_size: int = 48, 
                         padding: int = 16) -> Image.Image:
        """Render text using the given font file into an RGB image."""
        # Load font
        try:
            font = ImageFont.truetype(font_path, font_size, 
                                    layout_engine=ImageFont.Layout.BASIC)
        except Exception:
            # Fallback without specifying layout_engine
            font = ImageFont.truetype(font_path, font_size)
        
        # Measure text bbox
        dummy_img = Image.new("RGB", (1, 1), "white")
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        
        # Create final image
        img_w = width + padding * 2
        img_h = height + padding * 2
        img = Image.new("RGB", (img_w, img_h), "white")
        draw = ImageDraw.Draw(img)
        draw.text((padding, padding), text, fill="black", font=font)
        
        return img
    
    @staticmethod
    def resize_image_to_height(img: Image.Image, target_height: int) -> Image.Image:
        """Resize image to target height while maintaining aspect ratio."""
        if target_height <= 0 or img.height == target_height:
            return img
        
        scale = target_height / max(1, img.height)
        new_width = max(1, int(img.width * scale))
        return img.resize((new_width, target_height), Image.LANCZOS)
    
    @staticmethod
    def image_to_data_uri(img: Image.Image) -> str:
        """Convert PIL Image to data URI."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"


class DataFrameBuilder:
    """Handles DataFrame construction and manipulation."""
    
    def __init__(self):
        self.unicode_helper = UnicodeHelper()
    
    def build_comparison_dataframe(self, font_infos: List[FontInfo]) -> pd.DataFrame:
        """Build comparison DataFrame from font information."""
        # Get union of all codepoints
        all_codepoints = set()
        for font_info in font_infos:
            all_codepoints.update(font_info.codepoints)
        
        # Build base DataFrame
        sorted_codepoints = sorted(all_codepoints)
        df = pd.DataFrame({
            "Unicode": [self.unicode_helper.format_unicode(cp) for cp in sorted_codepoints],
            "Character": [self.unicode_helper.safe_chr(cp) for cp in sorted_codepoints],
            "_cp": sorted_codepoints,  # helper for sorting and joins
        })
        
        # Add per-font support columns
        for font_info in font_infos:
            mask = df["_cp"].isin(font_info.codepoints)
            df[font_info.name] = mask.map(lambda x: "T" if x else "F")
        
        # Add coverage count column
        font_columns = [info.name for info in font_infos]
        df["CoverageCount"] = df[font_columns].apply(
            lambda row: sum(1 for v in row if v == "T"), axis=1
        )
        
        return df
    
    def filter_and_sort(self, df: pd.DataFrame, filter_text: str, sort_by: str,
                       sort_order: str, max_rows: int, diff_only: bool,
                       font_count: int) -> pd.DataFrame:
        """Filter and sort the DataFrame based on parameters."""
        filtered = df.copy()
        
        # Apply text filter
        if filter_text:
            filtered = self._apply_text_filter(filtered, filter_text.strip())
        
        # Apply differences filter
        if diff_only and font_count > 1:
            filtered = filtered[
                (filtered["CoverageCount"] > 0) & 
                (filtered["CoverageCount"] < font_count)
            ]
        
        # Apply sorting
        filtered = self._apply_sorting(filtered, sort_by, sort_order)
        
        # Limit rows for performance
        if max_rows and max_rows > 0:
            filtered = filtered.head(max_rows)
        
        # Reorder columns
        font_cols = [c for c in filtered.columns 
                    if c not in {"_cp", "Unicode", "Character", "CoverageCount"}]
        ordered_cols = ["Character", "Unicode", "CoverageCount"] + font_cols
        
        return filtered[ordered_cols].reset_index(drop=True)
    
    def _apply_text_filter(self, df: pd.DataFrame, text: str) -> pd.DataFrame:
        """Apply text-based filtering to DataFrame."""
        if not text:
            return df
        
        # Try to interpret as hex number
        try:
            cleaned = text.upper().replace("U+", "").replace("0X", "")
            codepoint = int(cleaned, 16)
            unicode_str = self.unicode_helper.format_unicode(codepoint)
            return df[df["Unicode"].str.contains(unicode_str, case=False, na=False)]
        except ValueError:
            # Search by substring in Unicode or exact character match
            return df[
                df["Unicode"].str.contains(text, case=False, na=False) |
                (df["Character"] == text)
            ]
    
    def _apply_sorting(self, df: pd.DataFrame, sort_by: str, sort_order: str) -> pd.DataFrame:
        """Apply sorting to DataFrame."""
        ascending = sort_order.lower().startswith("asc")
        
        if sort_by == "Unicode":
            return df.sort_values(by="_cp", ascending=ascending, kind="mergesort")
        elif sort_by in ["Character", "CoverageCount"]:
            return df.sort_values(by=sort_by, ascending=ascending, kind="mergesort")
        elif sort_by in df.columns:
            # Sort by font column
            key_col = f"__key_{sort_by}__"
            df[key_col] = (df[sort_by] == "T").astype(int)
            sorted_df = df.sort_values(
                by=[key_col, "_cp"], 
                ascending=[ascending, True], 
                kind="mergesort"
            )
            return sorted_df.drop(columns=[key_col])
        
        return df


class CSVExporter:
    """Handles CSV export functionality."""
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: str = "comparison.csv") -> str:
        """Export DataFrame to CSV with robust encoding handling."""
        tmpdir = tempfile.mkdtemp(prefix="fontmatrix_")
        csv_path = os.path.join(tmpdir, filename)
        
        try:
            # First attempt with UTF-8 BOM for Excel compatibility
            df.to_csv(csv_path, index=False, encoding="utf-8-sig", errors="replace")
        except UnicodeEncodeError:
            # Fallback: clean problematic characters
            df_safe = df.copy()
            if "Character" in df_safe.columns:
                df_safe["Character"] = df_safe["Character"].apply(
                    lambda x: (x.encode("utf-8", errors="replace").decode("utf-8") 
                             if isinstance(x, str) else x)
                )
            df_safe.to_csv(csv_path, index=False, encoding="utf-8-sig", errors="replace")
        
        return csv_path


class FontMatrixApp:
    """Main application class that coordinates all components."""
    
    def __init__(self):
        self.font_processor = FontProcessor()
        self.dataframe_builder = DataFrameBuilder()
        self.csv_exporter = CSVExporter()
        self.image_renderer = ImageRenderer()
    
    def process_fonts(self, file_paths: List[str], progress: gr.Progress) -> Tuple[pd.DataFrame, List[FontInfo]]:
        """Process uploaded font files and return DataFrame and font info."""
        if not file_paths:
            raise ValueError("No font files provided.")
        
        if len(file_paths) > MAX_FONTS:
            raise ValueError(f"Please upload at most {MAX_FONTS} font files.")
        
        # Validate files
        for path in file_paths:
            if not self.font_processor.is_valid_font_extension(path):
                raise ValueError(f"Unsupported file type: {Path(path).name}. "
                               f"Allowed: {', '.join(SUPPORTED_EXTENSIONS)}")
            if not os.path.exists(path):
                raise ValueError(f"File not found: {path}")
        
        # Create unique names and process fonts
        names = self.font_processor.create_unique_names(file_paths)
        font_infos = []
        
        for i, (name, path) in enumerate(zip(names, file_paths), start=1):
            progress((i - 1) / len(file_paths), desc=f"Parsing {name}")
            try:
                codepoints = self.font_processor.extract_codepoints(path)
                font_infos.append(FontInfo(name=name, path=path, codepoints=codepoints))
            except Exception as e:
                raise ValueError(f"Error parsing {name}: {e}")
        
        progress(0.75, desc="Building comparison table")
        df = self.dataframe_builder.build_comparison_dataframe(font_infos)
        progress(0.9, desc="Finalizing table")
        
        return df, font_infos
    
    def compare_fonts(self, files: Optional[List[Union[str, gr.File]]], filter_text: str,
                     sort_by: str, sort_order: str, max_rows: int, diff_only: bool = False,
                     progress: gr.Progress = gr.Progress(track_tqdm=True)):
        """Main handler for font comparison."""
        start_time = time.time()
        
        try:
            # Normalize file paths
            file_paths = self._normalize_file_paths(files)
            if not file_paths:
                raise ValueError("Please upload at least one .ttf or .otf font file.")
            
            # Process fonts
            df, font_infos = self.process_fonts(file_paths, progress)
            font_names = [info.name for info in font_infos]
            
            # Filter and sort
            filtered_df = self.dataframe_builder.filter_and_sort(
                df, filter_text, sort_by, sort_order, max_rows, diff_only, len(font_names)
            )
            
            # Generate summary
            elapsed = time.time() - start_time
            summary = self._generate_summary(font_names, len(df), len(filtered_df), elapsed)
            
            # Export CSV
            csv_path = self.csv_exporter.export_to_csv(filtered_df)
            
            # Update sort options
            sort_choices = ["Unicode", "Character", "CoverageCount"] + font_names
            
            return filtered_df, summary, csv_path, gr.update(choices=sort_choices)
            
        except Exception as e:
            return (
                pd.DataFrame(),
                f"❗ Error: {str(e)}",
                None,
                gr.update(),
            )
    
    def generate_preview_html(self, files: Optional[List[Union[str, gr.File]]], 
                            text: str, height_px: int) -> str:
        """Generate HTML preview of text rendered in each font."""
        try:
            if not files:
                return ""
            
            # Normalize file paths and text
            file_paths = self._normalize_file_paths(files)
            sample_text = (text or "").strip() or DEFAULT_SAMPLE_TEXT
            names = self.font_processor.create_unique_names(file_paths)
            
            # Generate preview rows
            rows_html = []
            for name, path in zip(names, file_paths):
                if not self.font_processor.is_valid_font_extension(path):
                    continue
                
                try:
                    img = self.image_renderer.render_text_image(sample_text, path)
                    img = self.image_renderer.resize_image_to_height(img, int(height_px or 96))
                    uri = self.image_renderer.image_to_data_uri(img)
                    
                    row = (
                        f'<div class="font-row">'
                        f'<div class="font-caption">{name}</div>'
                        f'<img src="{uri}" alt="{name}" '
                        f'style="height:{int(height_px or 96)}px;max-width:100%;display:block;" />'
                        f'</div>'
                    )
                    rows_html.append(row)
                except Exception:
                    # Skip fonts that fail to render
                    continue
            
            # Generate CSS and HTML
            styles = (
                "<style>"
                "#font-previews .font-row{display:flex;align-items:center;gap:16px;"
                "padding:8px 0;border-bottom:1px solid rgba(128,128,128,.3);}"
                "#font-previews .font-caption{width:260px;font-family:monospace;"
                "font-size:13px;word-break:break-all;opacity:.85;}"
                "</style>"
            )
            
            return f"{styles}<div id='font-previews'>{''.join(rows_html)}</div>"
            
        except Exception:
            return ""
    
    def _normalize_file_paths(self, files: Optional[List[Union[str, gr.File]]]) -> List[str]:
        """Convert Gradio file objects to file paths."""
        if not files:
            return []
        
        file_paths = []
        for file in files:
            if isinstance(file, str):
                file_paths.append(file)
            elif hasattr(file, "name"):
                file_paths.append(getattr(file, "name"))
        
        return file_paths
    
    def _generate_summary(self, font_names: List[str], total_rows: int, 
                         shown_rows: int, elapsed: float) -> str:
        """Generate summary text for the UI."""
        return (
            f"Processed {len(font_names)} fonts in {elapsed:.1f}s. "
            f"Showing {shown_rows:,} of {total_rows:,} rows.\n\n"
            f"Fonts: " + ", ".join(font_names)
        )


class UIBuilder:
    """Builds the Gradio interface."""
    
    def __init__(self, app: FontMatrixApp):
        self.app = app
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        with gr.Blocks(title="Font Matrix") as demo:
            gr.Markdown("# Font Matrix\nUpload up to 5 font files (.ttf, .otf) and compare their character coverage.")
            
            # Input section
            with gr.Row():
                file_input = self._create_file_input()
                controls_column = self._create_controls_column()
            
            # Action buttons
            with gr.Row():
                compare_btn = gr.Button("Compare Fonts", variant="primary")
                clear_btn = gr.ClearButton([
                    file_input, controls_column["sample_text"], controls_column["filter_tb"],
                    controls_column["diff_only"], controls_column["sort_by"], 
                    controls_column["sort_order"], controls_column["max_rows"]
                ])
            
            # Output section
            outputs = self._create_output_section()
            
            # Set up event handlers
            self._setup_event_handlers(
                compare_btn, file_input, controls_column, outputs
            )
            
        return demo
    
    def _create_file_input(self) -> gr.File:
        """Create file input component."""
        return gr.File(
            label="Font files (.ttf, .otf)",
            file_count="multiple",
            file_types=[".ttf", ".otf"],
        )
    
    def _create_controls_column(self) -> Dict[str, gr.Component]:
        """Create controls column with all input components."""
        with gr.Column():
            sample_text = gr.Textbox(
                label="Sample text (preview across fonts)",
                placeholder="Type a sentence to preview, e.g. The quick brown fox 一二三四",
                value=DEFAULT_SAMPLE_TEXT,
            )
            preview_height = gr.Slider(
                label="Preview height (px)", 
                minimum=24, maximum=160, step=4, value=96
            )
            filter_tb = gr.Textbox(
                label="Search / Filter (character, substring, or hex like 4E00 or U+4E00)",
                placeholder="e.g., 一 or 4E00",
            )
            diff_only = gr.Checkbox(label="Show only differences", value=False)
            sort_by = gr.Dropdown(
                label="Sort by",
                choices=["Unicode", "Character", "CoverageCount"],
                value="Unicode",
            )
            sort_order = gr.Radio(
                label="Sort order",
                choices=["Ascending", "Descending"],
                value="Ascending",
                interactive=True,
            )
            max_rows = gr.Slider(
                label="Max rows to display (for performance)",
                minimum=100, maximum=50000, step=100, value=5000,
            )
        
        return {
            "sample_text": sample_text,
            "preview_height": preview_height,
            "filter_tb": filter_tb,
            "diff_only": diff_only,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "max_rows": max_rows,
        }
    
    def _create_output_section(self) -> Dict[str, gr.Component]:
        """Create output section components."""
        df_out = gr.Dataframe(
            headers=["Character", "Unicode", "CoverageCount"],
            interactive=False,
            elem_id="comparison-table",
            label="Comparison Table",
        )
        note_md = gr.Markdown()
        csv_out = gr.File(label="Download CSV")
        
        gr.Markdown("### Font Previews")
        preview_html = gr.HTML()
        
        return {
            "df_out": df_out,
            "note_md": note_md,
            "csv_out": csv_out,
            "preview_html": preview_html,
        }
    
    def _setup_event_handlers(self, compare_btn: gr.Button, file_input: gr.File,
                            controls: Dict[str, gr.Component], 
                            outputs: Dict[str, gr.Component]):
        """Set up all event handlers for the interface."""
        # Main comparison handler
        compare_inputs = [
            file_input, controls["filter_tb"], controls["sort_by"],
            controls["sort_order"], controls["max_rows"], controls["diff_only"]
        ]
        compare_outputs = [
            outputs["df_out"], outputs["note_md"], outputs["csv_out"], controls["sort_by"]
        ]
        
        compare_btn.click(
            fn=self.app.compare_fonts,
            inputs=compare_inputs,
            outputs=compare_outputs,
        )
        
        # Preview generation handler
        preview_inputs = [file_input, controls["sample_text"], controls["preview_height"]]
        preview_outputs = [outputs["preview_html"]]
        
        compare_btn.click(
            fn=self.app.generate_preview_html,
            inputs=preview_inputs,
            outputs=preview_outputs,
        )
        
        # Live filtering/sorting handlers
        filter_components = [
            controls["filter_tb"], controls["diff_only"], controls["sort_by"],
            controls["sort_order"], controls["max_rows"]
        ]
        
        for component in filter_components:
            component.change(
                fn=self.app.compare_fonts,
                inputs=compare_inputs,
                outputs=compare_outputs,
            )
        
        # Preview update handlers
        preview_components = [file_input, controls["sample_text"], controls["preview_height"]]
        
        for component in preview_components:
            component.change(
                fn=self.app.generate_preview_html,
                inputs=preview_inputs,
                outputs=preview_outputs,
            )


def main():
    """Main entry point for the application."""
    app = FontMatrixApp()
    ui_builder = UIBuilder(app)
    demo = ui_builder.create_interface()
    demo.launch()


if __name__ == "__main__":
    main()