import logging
from datetime import datetime
from sqlalchemy.orm import Session
from config.database import engine, SessionLocal
from config.models import Log
from colorama import init, Fore, Back, Style

init(autoreset=True)

class DBHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        db = SessionLocal()
        db_log = Log(
            timestamp=datetime.now(),
            level=record.levelname,
            message=log_entry
        )
        db.add(db_log)
        db.commit()
        db.close()

class ColorFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.DEBUG:
            record.msg = Fore.CYAN + str(record.msg)
        elif record.levelno == logging.INFO:
            record.msg = Fore.GREEN + str(record.msg)
        elif record.levelno == logging.WARNING:
            record.msg = Fore.YELLOW + str(record.msg)
        elif record.levelno == logging.ERROR:
            record.msg = Fore.RED + str(record.msg)
        elif record.levelno == logging.CRITICAL:
            record.msg = Back.RED + Fore.WHITE + str(record.msg)
        return super().format(record)

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # File handler for the model API
    model_file_handler = logging.FileHandler("model_api.log")
    model_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levellevelname)s - %(message)s')
    model_file_handler.setFormatter(model_formatter)
    model_file_handler.setLevel(logging.DEBUG)

    # Console handler with color formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    console_handler.setLevel(logging.DEBUG)

    # DB handler
    db_handler = DBHandler()
    db_handler.setFormatter(logging.Formatter('%(message)s'))
    db_handler.setLevel(logging.DEBUG)

    # Add handlers to the logger
    logger.addHandler(model_file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(db_handler)
    
    return logger
