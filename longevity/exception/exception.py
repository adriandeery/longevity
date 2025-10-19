# longevity/exception/exception.py
import sys
from longevity.logging.logger import logger


class LongevityException(Exception):
    """Base exception for the longevity pipeline."""

    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(
            error_message, error_detail
        )
        logger.error(self.error_message)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: sys) -> str:
        """Generate detailed error message with traceback info."""
        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is None:
            return f"Error: {error_message}"

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            f"Error occurred in script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"error message: [{error_message}]"
        )

    def __str__(self):
        return self.error_message


class DataIngestionError(LongevityException):
    """Exception raised for errors in data ingestion."""

    pass


class PreprocessingError(LongevityException):
    """Exception raised for errors in preprocessing."""

    pass


class ModelError(LongevityException):
    """Exception raised for errors in model operations."""

    pass


class APIError(LongevityException):
    """Exception raised for API-related errors."""

    pass
