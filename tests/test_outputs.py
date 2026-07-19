from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from logging_utils.outcomes import OutputLogger
from logging_utils.writer import read_jsonl


class OutputLoggerTests(unittest.TestCase):
    def test_response_is_the_primary_output_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "outputs.jsonl"
            logger = OutputLogger(path)
            try:
                logger.log(
                    context={"example_id": "example-1", "prompt": "hello"},
                    final_text="clean answer",
                    raw_response="<start>clean answer<end>",
                    final_token_ids=[1, 2],
                    input_token_count=3,
                    rendered_prompt="formatted hello",
                )
            finally:
                logger.close()

            record = list(read_jsonl(path))[0]
            self.assertEqual(record["response"], "clean answer")
            self.assertEqual(record["final_text"], "clean answer")
            self.assertEqual(record["raw_response"], "<start>clean answer<end>")
            self.assertEqual(record["rendered_prompt"], "formatted hello")


if __name__ == "__main__":
    unittest.main()
