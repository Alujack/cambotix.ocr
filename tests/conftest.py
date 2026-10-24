from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest

SYNTHETIC_RESPONSES: dict[str, tuple[str, float | None]] = {
    "id_number": ("000000001", 0.97),
    "full_name_km": ("គោត្តនាមនិងនាម: គំរូ សាកល្បង", 0.93),
    "full_name_latin": ("KUMRU SAKALBANG", 0.95),
    "date_of_birth": ("ថ្ងៃខែឆ្នាំកំណើត: 01.01.1990", 0.91),
    "gender": ("ភេទ: ប្រុស", 0.9),
}

# Values that must never appear in logs.
SENSITIVE_VALUES = ["000000001", "គំរូ", "KUMRU", "SAKALBANG", "1990"]


class NetworkAccessError(AssertionError):
    pass


@pytest.fixture
def no_network(monkeypatch) -> Iterator[list[tuple[str, object]]]:
    """Fail on any outbound connection or DNS lookup; record attempts."""
    attempts: list[tuple[str, object]] = []

    def guard(name: str):
        def blocked(*args, **kwargs):
            attempts.append((name, args[1:] if args else kwargs))
            raise NetworkAccessError(f"network access attempted via {name}")
        return blocked

    monkeypatch.setattr(socket.socket, "connect", guard("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", guard("socket.connect_ex"))
    monkeypatch.setattr(socket, "getaddrinfo", guard("getaddrinfo"))
    monkeypatch.setattr(socket, "create_connection", guard("create_connection"))
    yield attempts
