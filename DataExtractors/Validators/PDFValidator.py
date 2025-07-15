import os
import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError
from Contracts.IHelperMethod import IHelperMethod

class PDFValidator(IHelperMethod):
    def validate(self, pdf_path:str) -> tuple[bool, str]:
        """
               Checks if the PDF file exists, is valid and is readable.
               Args:
                   pdf_path: Path to the PDF file
               Returns:
                   tuple: (bool, str) - (True if valid PDF, error message if invalid)
        """
        if not os.path.exists(pdf_path):
            return False, 'File does not exist'

        if not os.access(pdf_path, os.R_OK):
            return False, 'No read permissions'

        if not pdf_path.lower().endswith('.pdf'):
            return False, 'Not a PDF file'

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if not pdf.pages:
                    return False, 'Empty PDF (no pages)'

                # Test text extraction from first page
                _ = pdf.pages[0].extract_text()
                return True, 'Valid PDF file'

        except PDFSyntaxError as e:
            return False, f'Invalid PDF structure: {str(e)}'
        except Exception as e:
            return False, f'Error validating PDF: {str(e)}'
