from __future__ import annotations

from typing import Any, Protocol


class BackendNotAvailable(RuntimeError):
    pass


class InferBackend(Protocol):
    name: str

    def infer(self, tensor: Any) -> Any: ...

    def close(self) -> None: ...
