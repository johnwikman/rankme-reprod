import logging
import sys

LOG_LEVELS = [logging.INFO, logging.DEBUG]
LOG_FMT = logging.Formatter(
    "[%(asctime)s %(name)s:%(lineno)d %(levelname)s]: %(message)s"
)

def init_logging(verbosity=0, logfile=None, log_stderr=False):
    # Setup the root logger
    logging.getLogger().setLevel(LOG_LEVELS[min(verbosity, len(LOG_LEVELS) - 1)])
    if log_stderr:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(LOG_FMT)
        logging.getLogger().addHandler(stderr_handler)
    if logfile is not None:
        logfile_handler = logging.FileHandler(logfile)
        logfile_handler.setFormatter(LOG_FMT)
        logging.getLogger().addHandler(logfile_handler)


