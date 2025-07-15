import pdfplumber
from Contracts.IDataExtractor import IDataExtractor
from Contracts.IHelperMethod import IHelperMethod

class PdfExtractor(IDataExtractor):
    def __init__(self, validator : IHelperMethod):
        self.validator = validator

    def extract_text_from_path(self, pdf_path : str) -> str:
        """
        Extracts text from a PDF file after validation.
        Args:
            pdf_path: Path to the PDF file
        Returns:
            Extracted text as string
        Raises:
            ValueError: If the file is invalid
            RuntimeError: If text extraction fails
        """
        isvalidpdf, message = self.validator.validate(pdf_path)
        if not isvalidpdf:
            raise ValueError(message)

        try:
            text = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # Only add if text was extracted
                        text.append(page_text)

            return '\n'.join(text).strip() if text else ""

        except Exception as e:
            raise RuntimeError(message + f' Failed to extract text : {e}')

    def extract_text_from_pdf_file(self, pdf_file):
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    pdf_txt = page.extract_text()
                    if pdf_txt:
                        text += pdf_txt + "\n"

            return text

        except Exception as e:
            raise RuntimeError(f' Failed to extract text : {e}')
