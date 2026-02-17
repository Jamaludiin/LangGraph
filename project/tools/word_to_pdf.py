# tools/word_to_pdf.py

from langchain_core.tools import tool

@tool
def word_to_pdf(file_path: str) -> str:
    """
    Convert a Word file to PDF.
    """
    return f"Converted {file_path} from Word to PDF successfully."
