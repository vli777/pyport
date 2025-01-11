# tests/test_logger.py

import unittest
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from src.utils.logger import logger


class TestLogger(unittest.TestCase):

    def test_logger_exists(self):
        self.assertIsInstance(logger, logging.Logger)

    def test_logger_level(self):
        # Assuming the logger level is set to DEBUG
        self.assertEqual(logger.level, logging.DEBUG)

    def test_logger_handlers(self):
        # Check if logger has at least two handlers (console and file)
        self.assertTrue(len(logger.handlers) >= 2)

    def test_logger_output(self):
        # This test captures log output and verifies it
        with self.assertLogs("project_logger", level="INFO") as log:
            logger.info("Test log message.")
            self.assertIn("Test log message.", log.output[0])


if __name__ == "__main__":
    unittest.main()
