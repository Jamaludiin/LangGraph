# tools/__init__.py

from .pdf_to_word import pdf_to_word
from .pdf_to_image import pdf_to_image
from .md_to_pdf import md_to_pdf
from .word_to_pdf import word_to_pdf

ALL_TOOLS = [
    pdf_to_word,
    pdf_to_image,
    md_to_pdf,
    word_to_pdf
]
