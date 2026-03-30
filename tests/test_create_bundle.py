import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scripts.models.create_bundle as cb

def test_resolve_bundle_filename_adds_extension_if_missing():
    result = cb._resolve_bundle_filename("foo", "20210101")
    assert result == "foo.mag"

def test_resolve_bundle_filename_keeps_extension_if_present():
    result = cb._resolve_bundle_filename("bar.mag", "20210101")
    assert result == "bar.mag"

def test_resolve_bundle_filename_uses_timestamp_when_none():
    result = cb._resolve_bundle_filename(None, "20210101")
    assert result == "magenta_20210101.mag"
