# tools/pdf_to_word.py

from langchain_core.tools import tool

@tool
def pdf_to_word(file_path: str) -> str:
    """
    Convert a PDF file to Word format.
    """
    # Simulated conversion
    return f"Converted {file_path} from PDF to Word successfully."
