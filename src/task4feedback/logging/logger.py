import logging
import logging.config
from pythonjsonlogger import jsonlogger
import os


ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "1")
ENABLE_LOGGING = ENABLE_LOGGING in ("true", "True", "TRUE", "1", "t", "y", "yes")
if ENABLE_LOGGING:
    if not os.path.exists("logs"):
        os.makedirs("logs")

    LOGFILE = os.path.join("logging.conf")
    print("Notice: Logging enabled")

    logging.config.fileConfig(LOGFILE)

    runtime = logging.getLogger("runtime")
    resource = logging.getLogger("resources")
    state = logging.getLogger("state")
    data = logging.getLogger("data")
    stats = logging.getLogger("stats")
    mapping = logging.getLogger("mapping")
    launching = logging.getLogger("launching")
