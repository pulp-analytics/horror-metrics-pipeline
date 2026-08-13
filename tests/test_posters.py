"""Unit tests for utils/posters.py's cache-tier logic -- no real network
calls. Confirms the local-disk-cache short-circuit (the common case once
any script has already fetched a poster) never touches S3 or TMDB."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from utils.posters import fetch_poster_file  # noqa: E402


class ExplodingSession:
    """A requests.Session stand-in that fails the test if .get() is ever
    called -- used to prove the local-cache-hit path never reaches TMDB."""

    def get(self, *args, **kwargs):
        raise AssertionError("fetch_poster_file hit the network despite dest already existing")


def test_local_cache_hit_skips_network_entirely(tmp_path):
    dest = tmp_path / "12345.jpg"
    dest.write_bytes(b"already cached")

    ok = fetch_poster_file(ExplodingSession(), "/some/poster.jpg", dest)

    assert ok is True
    assert dest.read_bytes() == b"already cached"  # untouched, not re-fetched


def test_local_cache_hit_skips_s3_too(tmp_path):
    dest = tmp_path / "999.jpg"
    dest.write_bytes(b"cached")

    # even with an S3 bucket configured, an existing local file should
    # short-circuit before any S3 or TMDB call is attempted
    ok = fetch_poster_file(ExplodingSession(), "/x.jpg", dest,
                            s3_bucket="some-bucket", s3_prefix="some-prefix")

    assert ok is True


def test_no_s3_bucket_means_no_s3_attempt(monkeypatch, tmp_path):
    # with s3_bucket="" (the default), boto3 should never even be imported --
    # confirms this repo's sample data truly needs zero AWS involvement
    import builtins
    real_import = builtins.__import__

    def guard(name, *a, **kw):
        if name == "boto3":
            raise AssertionError("boto3 should not be imported when s3_bucket is empty")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", guard)

    dest = tmp_path / "1.jpg"

    class FailingSession:
        def get(self, *a, **kw):
            class R:
                status_code = 404
                content = b""
            return R()

    ok = fetch_poster_file(FailingSession(), "/x.jpg", dest)
    assert ok is False
