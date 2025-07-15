import os
import zipfile
import docx2txt
from Contracts.IHelperMethod import IHelperMethod

class WordValidator(IHelperMethod):
    def validate(self, word_path : str) -> tuple[bool,str]:
        """
            Checks if the Word file exists, is valid and is readable.
            Args:
                word_path: Path to the Word file
            Returns:
                tuple: (bool, str) - (True if valid Word file, error message if invalid)
        """
        if not os.path.exists(word_path):
            return False, 'File does not exist'

        if not os.access(word_path, os.R_OK):
            return False, 'No read permissions'

        if not word_path.lower().endswith(('.doc', '.docx')):
            return False, 'Not a Word document'

        try:
            # Test processing the file
            _ = docx2txt.process(word_path)
            return True, 'Valid Word document'
        except zipfile.BadZipFile:
            return False, 'Corrupted Word file'
        except Exception as e:
            return False, f'Error validating Word file: {str(e)}'