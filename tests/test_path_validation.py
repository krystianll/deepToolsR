import os

import pytest

from deeptoolsr.path_validation import (
    atomic_output_path,
    temporary_path_for,
    validate_input_output_paths,
)
from deeptoolsr.parserCommon import writableFile


def test_rejects_relative_and_symlink_input_aliases(tmp_path, monkeypatch):
    source = tmp_path / "input.dat"
    source.write_text("data")
    link = tmp_path / "link.dat"
    link.symlink_to(source)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="aliases input"):
        validate_input_output_paths(["input.dat"], [link])


def test_rejects_existing_hardlink_alias(tmp_path):
    source = tmp_path / "input.dat"
    source.write_text("data")
    link = tmp_path / "hardlink.dat"
    os.link(source, link)
    with pytest.raises(ValueError, match="aliases input"):
        validate_input_output_paths([source], [link])


def test_rejects_duplicate_outputs_but_allows_unrelated_existing_output(tmp_path):
    source = tmp_path / "input.dat"
    source.write_text("input")
    output = tmp_path / "output.dat"
    output.write_text("old output")
    validate_input_output_paths([source], [output])
    with pytest.raises(ValueError, match="same file"):
        validate_input_output_paths([source], [output, output])


def test_preflight_rejects_missing_parent_and_directory_destination(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        validate_input_output_paths([], [tmp_path / 'missing' / 'output.dat'])
    with pytest.raises(ValueError, match='is a directory'):
        validate_input_output_paths([], [tmp_path])


def test_writable_file_check_does_not_touch_existing_file(tmp_path):
    path = tmp_path / "existing.dat"
    path.write_text("keep me")
    assert writableFile(str(path)) == str(path)
    assert path.read_text() == "keep me"


def test_temporary_path_is_unique_and_cleaned(tmp_path):
    destination = tmp_path / "result.gz"
    with temporary_path_for(destination, suffix=".body.gz") as first:
        with temporary_path_for(destination, suffix=".body.gz") as second:
            assert first != second
            assert os.path.dirname(first) == str(tmp_path)
            assert os.path.exists(first)
            assert os.path.exists(second)
    assert not os.path.exists(first)
    assert not os.path.exists(second)


def test_atomic_output_preserves_old_file_on_failure_and_replaces_on_success(tmp_path):
    destination = tmp_path / "result.dat"
    destination.write_text("old")
    with pytest.raises(RuntimeError):
        with atomic_output_path(destination) as temporary:
            with open(temporary, "w") as handle:
                handle.write("partial")
            raise RuntimeError("injected failure")
    assert destination.read_text() == "old"

    with atomic_output_path(destination) as temporary:
        with open(temporary, "w") as handle:
            handle.write("new")
    assert destination.read_text() == "new"
