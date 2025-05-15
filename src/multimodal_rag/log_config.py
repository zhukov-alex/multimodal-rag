import logging
import sys
from pythonjsonlogger import jsonlogger

logger = logging.getLogger("multimodal_rag")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s",
    rename_fields={"levelname": "level", "asctime": "timestamp"}
)
handler.setFormatter(formatter)
logger.addHandler(handler)
