import docx2txt
import pdfplumber
from Contracts.IDataExtractor import IDataExtractor
from Contracts.IHelperMethod import IHelperMethod
import aspose.words as aw

class WordExtractor(IDataExtractor):
    def __init__(self, validator : IHelperMethod):
        self.validator = validator

    @staticmethod
    def convert_word_to_pdf_aspose(word_path, pdf_path=None):
        """Convert Word to PDF using Aspose.Words"""
        if pdf_path is None:
            pdf_path = word_path.replace('.docx', '.pdf').replace('.doc', '.pdf')
        # Load the document
        doc = aw.Document(word_path)
        # Save as PDF
        doc.save(pdf_path)
        return pdf_path

    def extract_text(self, docx_path: str) -> str:
        """
               Extracts text from a DOCX file after validation and Conversion into a PDF.
               Args:
                   docx_path: Path to the DOCX file
               Returns:
                   Extracted text as string
               Raises:
                   ValueError: If the file is invalid
                   RuntimeError: If text extraction fails
        """
        is_valid_word , message = self.validator.validate(docx_path)
        if not is_valid_word:
            raise ValueError(message)

        try:
            text_from_word = []
            pdf_path = self.convert_word_to_pdf_aspose(docx_path)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # Only add if text was extracted
                        text_from_word.append(page_text)

            return '\n'.join(text_from_word).strip() if text_from_word else ""

        except Exception as e:
            raise RuntimeError(message + f' Failed to extract text. {e}')
