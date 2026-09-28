import pytest

from physical_ai.io import sha256_file
from physical_ai.tau_export import audit_tau_libero, verify_checksums


def test_manifest_rejects_corruption(tmp_path):
    file = tmp_path / "config.json"
    file.write_text("{}")
    digest = sha256_file(file)
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  config.json\n")
    assert verify_checksums(tmp_path) == {"config.json": digest}
    file.write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksums(tmp_path)


def test_manifest_rejects_duplicate_and_escape(tmp_path):
    file = tmp_path / "config.json"
    file.write_text("{}")
    digest = sha256_file(file)
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  config.json\n" * 2)
    with pytest.raises(ValueError, match="repeated path"):
        verify_checksums(tmp_path)
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  ../outside\n")
    with pytest.raises(ValueError, match="escaping"):
        verify_checksums(tmp_path)


def test_manifest_must_match_pinned_export(tmp_path):
    (tmp_path / "SHA256SUMS").write_text("")
    with pytest.raises(ValueError, match="empty"):
        verify_checksums(tmp_path)
    with pytest.raises(ValueError, match="pinned export"):
        audit_tau_libero(tmp_path)
