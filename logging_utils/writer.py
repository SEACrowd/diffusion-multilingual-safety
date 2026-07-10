"""JSON and JSONL persistence shared by telemetry components."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._output: TextIO | None = self.path.open("a", encoding="utf-8")

    def write(self, value: Any) -> None:
        if self._output is None:
            raise RuntimeError(f"Cannot write to closed JSONL writer for {self.path}")
        self._output.write(json.dumps(value, ensure_ascii=False) + "\n")
        self._output.flush()

    def close(self) -> None:
        if self._output is not None:
            self._output.close()
            self._output = None

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def append_jsonl(path: str | Path, value: Any) -> None:
    with JsonlWriter(path) as writer:
        writer.write(value)


def write_jsonl(path: str | Path, values: Iterable[Any]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for value in values:
                output.write(json.dumps(value, ensure_ascii=False) + "\n")
                count += 1
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield dict(value)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(destination)
