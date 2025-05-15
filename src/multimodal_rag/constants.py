SUPPORTED_TEXT = {".txt", ".md"}
SUPPORTED_DOCS = {".pdf", ".docx"}
SUPPORTED_CODE = {".py", ".js", ".ts", ".java", ".go", ".rs"}
SUPPORTED_HTML = {".html", ".htm"}
SUPPORTED_NOTEBOOK = {".ipynb"}
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

SUPPORTED_ALL = (
    SUPPORTED_TEXT
    | SUPPORTED_DOCS
    | SUPPORTED_CODE
    | SUPPORTED_HTML
    | SUPPORTED_NOTEBOOK
    | SUPPORTED_IMAGES
)

SUPPORTED_ARCHIVES = {".zip", ".tar.gz", ".tar"}
KNOWN_BUT_UNSUPPORTED = {".rar", ".7z"}

__all__ = [
    "SUPPORTED_TEXT",
    "SUPPORTED_DOCS",
    "SUPPORTED_CODE",
    "SUPPORTED_HTML",
    "SUPPORTED_NOTEBOOK",
    "SUPPORTED_IMAGES",
    "SUPPORTED_ALL",
    "SUPPORTED_ARCHIVES",
    "KNOWN_BUT_UNSUPPORTED",
]
