from __future__ import annotations

import os

from qmt_agent_trader.persistence.provenance import fingerprint_path_tree


def test_same_size_replacement_with_preserved_mtime_changes_fingerprint(
    tmp_path,
) -> None:
    path = tmp_path / "registry.json"
    path.write_bytes(b"AAAA")
    stat = path.stat()
    first = fingerprint_path_tree(path)

    path.write_bytes(b"BBBB")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = fingerprint_path_tree(path)

    assert first != second
