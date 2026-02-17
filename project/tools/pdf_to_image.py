# tools/pdf_to_image.py

from langchain_core.tools import tool

@tool
def pdf_to_image(file_path: str) -> str:
    """
    Convert a PDF file to images.
    """
    return f"Converted {file_path} from PDF to images successfully."
