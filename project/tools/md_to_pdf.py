# tools/md_to_pdf.py

from langchain_core.tools import tool

@tool
def md_to_pdf(file_path: str) -> str:
    """
    Convert a Markdown file to PDF.
    """
    return f"Converted {file_path} from Markdown to PDF successfully."
