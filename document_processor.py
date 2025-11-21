# document_processor.py
import fitz
from PIL import Image
import pytesseract
import io
import os
from typing import Any, Dict, List, Optional


class DocumentProcessor:
    """
    Clean, Pylance-friendly document processor for PDFs using PyMuPDF (fitz).
    Extracts:
      - Text chunks (page-level)
      - Table-like structured blocks (from text dict)
      - Images with OCR (pytesseract)
    """

    def __init__(self, pdf_path: str):
        self.pdf_path: str = pdf_path
        self.doc = fitz.open(pdf_path)

    # -----------------------------------------------------
    # TEXT EXTRACTION
    # -----------------------------------------------------
    def extract_text_chunks(self) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            try:
                raw_text = page.get_text("text")
            except Exception:
                raw_text = page.get_text()

            if isinstance(raw_text, str) and raw_text.strip():
                chunks.append({
                    "type": "text",
                    "content": raw_text,
                    "page": page_num + 1,
                    "source": f"Page {page_num + 1}"
                })

        return chunks

    # -----------------------------------------------------
    # TABLE / BLOCK EXTRACTION
    # -----------------------------------------------------
    def extract_tables(self) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            try:
                text_dict = page.get_text("dict")
            except Exception:
                continue

            if not isinstance(text_dict, dict):
                continue

            blocks = text_dict.get("blocks", [])
            if not isinstance(blocks, list):
                continue

            for block in blocks:
                if not isinstance(block, dict):
                    continue

                lines = block.get("lines", [])
                if not isinstance(lines, list):
                    continue

                # Heuristic: treat multi-line blocks as table-like
                if len(lines) > 2:
                    table_text = ""

                    for line in lines:
                        if not isinstance(line, dict):
                            continue

                        spans = line.get("spans", [])
                        if not isinstance(spans, list):
                            continue

                        for span in spans:
                            if not isinstance(span, dict):
                                continue

                            text = span.get("text", "")
                            if isinstance(text, str):
                                table_text += text + " "

                        table_text += "\n"

                    if table_text.strip():
                        tables.append({
                            "type": "table",
                            "content": table_text,
                            "page": page_num + 1,
                            "source": f"Table on Page {page_num + 1}"
                        })

        return tables

    # -----------------------------------------------------
    # IMAGE + OCR EXTRACTION
    # -----------------------------------------------------
    def extract_images_with_ocr(self, output_folder: Optional[str] = None) -> List[Dict[str, Any]]:
        # default images folder
        try:
            import config  # type: ignore
            default_folder = getattr(config, "IMAGES_DIR", "extracted_images")
        except Exception:
            default_folder = "extracted_images"

        # Choose output folder (ensure it's a string for Pylance)
        if not output_folder:
            output_folder = default_folder

        assert isinstance(output_folder, str), "output_folder must be a string"

        os.makedirs(output_folder, exist_ok=True)

        images_data: List[Dict[str, Any]] = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            try:
                image_list = page.get_images(full=True)  # type: ignore
            except TypeError:
                image_list = page.get_images()  # type: ignore
            except Exception:
                continue

            if not isinstance(image_list, list):
                continue

            for img_index, img in enumerate(image_list):
                if not isinstance(img, (list, tuple)) or len(img) == 0:
                    continue

                xref = img[0]

                try:
                    base_image = self.doc.extract_image(xref)  # type: ignore
                except Exception:
                    continue

                if not isinstance(base_image, dict):
                    continue

                image_bytes = base_image.get("image")
                if not isinstance(image_bytes, (bytes, bytearray)):
                    continue

                image_path = os.path.join(output_folder, f"page{page_num+1}_img{img_index+1}.png")

                try:
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                except Exception:
                    continue

                # OCR safely
                ocr_text = ""
                try:
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    raw = pytesseract.image_to_string(pil_img)
                    if isinstance(raw, str):
                        ocr_text = raw.strip()
                except Exception:
                    ocr_text = ""

                if ocr_text:
                    images_data.append({
                        "type": "image",
                        "content": ocr_text,
                        "page": page_num + 1,
                        "image_path": image_path,
                        "source": f"Image on Page {page_num + 1}"
                    })

        return images_data

    # -----------------------------------------------------
    # MAIN PIPELINE
    # -----------------------------------------------------
    def process_document(self) -> List[Dict[str, Any]]:
        print(f"Processing document: {self.pdf_path}")

        text_chunks = self.extract_text_chunks()
        tables = self.extract_tables()
        images = self.extract_images_with_ocr()

        print(
            f"Extracted: {len(text_chunks)} text chunks, "
            f"{len(tables)} tables, {len(images)} OCR images"
        )

        combined: List[Dict[str, Any]] = []
        combined.extend(text_chunks)
        combined.extend(tables)
        combined.extend(images)
        return combined

    def close(self) -> None:
        try:
            self.doc.close()
        except Exception:
            pass


if __name__ == "__main__":
    pdf = "sample.pdf"
    if os.path.exists(pdf):
        p = DocumentProcessor(pdf)
        chunks = p.process_document()
        print("Sample chunk:", chunks[0] if chunks else None)
        p.close()
    else:
        print(f"{pdf} not found.")
