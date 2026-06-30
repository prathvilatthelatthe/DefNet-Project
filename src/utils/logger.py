"""
DeforestNet - Logging System
Centralized logging configuration with console and file handlers.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Try to import colorlog for colored console output
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


class DeforestNetLogger:
    """Centralized logger for the DeforestNet project."""

    _loggers = {}

    @classmethod
    def get_logger(
        cls,
        name: str = "deforestnet",
        level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True
    ) -> logging.Logger:
        """
        Get or create a logger with the specified configuration.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            console_output: Whether to output to console

        Returns:
            Configured logger instance
        """
        # Return existing logger if already configured
        if name in cls._loggers:
            return cls._loggers[name]

        # Create new logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.handlers = []  # Clear any existing handlers

        # Format string
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"

        # Console handler with colors (if available)
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))

            if COLORLOG_AVAILABLE:
                color_format = colorlog.ColoredFormatter(
                    "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt=date_format,
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'bold_red',
                    }
                )
                console_handler.setFormatter(color_format)
            else:
                console_handler.setFormatter(
                    logging.Formatter(log_format, datefmt=date_format)
                )

            logger.addHandler(console_handler)

        # File handler with rotation
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(
                logging.Formatter(log_format, datefmt=date_format)
            )
            logger.addHandler(file_handler)

        # Prevent propagation to root logger
        logger.propagate = False

        # Store logger
        cls._loggers[name] = logger

        return logger

    @classmethod
    def setup_from_config(cls) -> logging.Logger:
        """
        Setup logger using configuration from config.py.

        Returns:
            Configured logger instance
        """
        try:
            from configs.config import LOGGING_CONFIG, LOGS_DIR

            return cls.get_logger(
                name="deforestnet",
                level=LOGGING_CONFIG.get("level", "INFO"),
                log_file=LOGGING_CONFIG.get("log_file", str(LOGS_DIR / "deforestnet.log")),
                console_output=LOGGING_CONFIG.get("console_output", True)
            )
        except ImportError:
            # Fallback if config not available
            return cls.get_logger()


# Convenience function
def get_logger(name: str = "deforestnet") -> logging.Logger:
    """
    Quick access to get a logger.

    Args:
        name: Logger name (will be prefixed with 'deforestnet.')

    Returns:
        Logger instance
    """
    full_name = f"deforestnet.{name}" if name != "deforestnet" else name
    return DeforestNetLogger.get_logger(full_name)


# Module-level logger for quick imports
logger = DeforestNetLogger.setup_from_config()


if __name__ == "__main__":
    # Test logging
    test_logger = get_logger("test")

    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")

    print("\n✓ Logging system test complete!")
