"""
Logging Configuration for Backend
Captures all logs for debugging via chatbot
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """
    Configure logging for the application

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """

    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

    # 1. Console Handler (simplified output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # 2. File Handler - All logs (rotating)
    all_logs_file = log_path / "backend.log"
    file_handler = RotatingFileHandler(
        all_logs_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # 3. Error Handler - Only errors and above
    error_logs_file = log_path / "errors.log"
    error_handler = RotatingFileHandler(
        error_logs_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)

    # Log startup
    logger.info("=" * 80)
    logger.info(f"Logging system initialized - Level: {log_level}")
    logger.info(f"Log directory: {log_path.absolute()}")
    logger.info("=" * 80)

    return logger


def get_logger(name: str):
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)


class ChatbotLogHandler(logging.Handler):
    """
    Custom handler that stores logs in memory for chatbot access
    Keeps last N log entries
    """

    def __init__(self, max_entries: int = 1000):
        super().__init__()
        self.max_entries = max_entries
        self.log_entries: list[dict[str, Any]] = []  # ✅ Ajout de l'annotation de type

    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "function": record.funcName,
                "line": record.lineno,
            }

            self.log_entries.append(log_entry)

            # Keep only last N entries
            if len(self.log_entries) > self.max_entries:
                self.log_entries = self.log_entries[-self.max_entries :]

        except Exception:
            self.handleError(record)

    def get_recent_logs(
        self, count: int = 50, level: str | None = None
    ):  # ✅ Correction Optional level
        """Get recent log entries, optionally filtered by level"""
        logs = self.log_entries

        if level:
            logs = [log for log in logs if log["level"] == level.upper()]

        return logs[-count:]

    def get_error_logs(self, count: int = 20):
        """Get recent error and warning logs"""
        error_logs = [
            log for log in self.log_entries if log["level"] in ["ERROR", "CRITICAL", "WARNING"]
        ]
        return error_logs[-count:]

    def clear(self):
        """Clear all stored logs"""
        self.log_entries = []


# Global instance for chatbot access
_chatbot_handler: ChatbotLogHandler | None = None


def setup_chatbot_logging():
    """Setup logging handler for chatbot access"""
    global _chatbot_handler

    if _chatbot_handler is None:
        _chatbot_handler = ChatbotLogHandler(max_entries=1000)
        _chatbot_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )
        _chatbot_handler.setFormatter(formatter)

        # Add to root logger
        logging.getLogger().addHandler(_chatbot_handler)

        logging.info("Chatbot log handler initialized")

    return _chatbot_handler


def get_chatbot_logs():
    """Get the chatbot log handler instance"""
    global _chatbot_handler
    if _chatbot_handler is None:
        setup_chatbot_logging()
    return _chatbot_handler
