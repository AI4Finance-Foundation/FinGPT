# PDFExtractor.py
import io
from typing import Tuple, Optional, Union, Dict, Any
import pdfplumber

# ----- Contracts (stubs to keep compatibility) -----
class IDataExtractor:
    def extract_text_from_path(self, pdf_path: str) -> str: ...
    def extract_text_from_pdf_file(self, pdf_file) -> str: ...

class IHelperMethod:
    def validate(self, path_or_file) -> Tuple[bool, str]: ...

# ---------------------------------------------------

class SimplePdfValidator(IHelperMethod):
    """Very lightweight validator; replace with your real one."""
    def validate(self, path_or_file) -> Tuple[bool, str]:
        try:
            if isinstance(path_or_file, (str, bytes, io.BufferedReader, io.BytesIO)):
                return True, "Valid PDF"
            return False, "Unsupported input type for PDF"
        except Exception as e:
            return False, f"Validation failed: {e}"

class PdfExtractor(IDataExtractor):
    def __init__(self, validator: Optional[IHelperMethod] = None):
        self.validator = validator or SimplePdfValidator()

    def _extract_text(self, pdf: pdfplumber.PDF) -> str:
        chunks = []
        for page in pdf.pages:
            txt = page.extract_text(x_tolerance=2, y_tolerance=2)
            if txt:
                chunks.append(txt.strip())
        return "\n\n".join(chunks).strip()

    def _extract_first_pages_text_and_meta(self, pdf: pdfplumber.PDF, pages: int = 2) -> Tuple[str, Dict[str, Any]]:
        meta = pdf.metadata or {}
        first_pages = pdf.pages[:max(1, pages)]
        t = "\n".join([(p.extract_text(x_tolerance=2, y_tolerance=2) or "") for p in first_pages])
        return t, meta

    # ---------- Basic APIs (existing) ----------
    def extract_text_from_path(self, pdf_path: str) -> str:
        is_valid, message = self.validator.validate(pdf_path)
        if not is_valid:
            raise ValueError(message)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return self._extract_text(pdf)
        except Exception as e:
            raise RuntimeError(f"{message} | Failed to extract text: {e}")

    def extract_text_from_pdf_file(self, pdf_file: Union[str, io.BytesIO]):
        is_valid, message = self.validator.validate(pdf_file)
        if not is_valid:
            raise ValueError(message)
        try:
            if isinstance(pdf_file, (io.BytesIO, io.BufferedReader)):
                pdf_file.seek(0)
                with pdfplumber.open(pdf_file) as pdf:
                    return self._extract_text(pdf)
            else:
                with pdfplumber.open(pdf_file) as pdf:
                    return self._extract_text(pdf)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text: {e}")

    # ---------- New combined APIs (text + header/meta) ----------
    def extract_text_and_header_from_path(self, pdf_path: str, header_pages: int = 2) -> Tuple[str, str, Dict[str, Any]]:
        """
        Returns: (full_text, first_pages_text, pdf_meta)
        """
        is_valid, message = self.validator.validate(pdf_path)
        if not is_valid:
            raise ValueError(message)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full = self._extract_text(pdf)
                first, meta = self._extract_first_pages_text_and_meta(pdf, header_pages)
                return full, first, meta
        except Exception as e:
            raise RuntimeError(f"{message} | Failed to extract text/meta: {e}")

    def extract_text_and_header_from_file(self, pdf_file: Union[str, io.BytesIO], header_pages: int = 2) -> Tuple[str, str, Dict[str, Any]]:
        """
        Returns: (full_text, first_pages_text, pdf_meta)
        """
        is_valid, message = self.validator.validate(pdf_file)
        if not is_valid:
            raise ValueError(message)
        try:
            if isinstance(pdf_file, (io.BytesIO, io.BufferedReader)):
                pdf_file.seek(0)
                with pdfplumber.open(pdf_file) as pdf:
                    full = self._extract_text(pdf)
                    first, meta = self._extract_first_pages_text_and_meta(pdf, header_pages)
                    return full, first, meta
            else:
                with pdfplumber.open(pdf_file) as pdf:
                    full = self._extract_text(pdf)
                    first, meta = self._extract_first_pages_text_and_meta(pdf, header_pages)
                    return full, first, meta
        except Exception as e:
            raise RuntimeError(f"Failed to extract text/meta: {e}")