import os
import tempfile

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


def _staged(directory, count):
    from deeptoolsr.path_validation import _reserve
    staged = []
    for index in range(count):
        destination = directory / f"out{index}.txt"
        destination.write_text(f"OLD{index}")
        temporary = _reserve(str(directory), ".tmp")
        with open(temporary, "w") as handle:
            handle.write("NEW")
        staged.append((temporary, str(destination)))
    return staged


def _contents(directory):
    return sorted((path.name, path.read_text()) for path in directory.iterdir())


def test_publish_rollback_survives_a_failed_restore(tmp_path, monkeypatch):
    from deeptoolsr import path_validation
    staged = _staged(tmp_path, 3)
    real_replace = os.replace
    calls = []

    def flaky(source, destination):
        calls.append(source)
        # 1-4 move out0/out1 aside and in; 5 is the last replace; 6 restores out1.
        if len(calls) == 5:
            raise PermissionError("final replace")
        if len(calls) == 6:
            raise PermissionError("restore out1")
        return real_replace(source, destination)

    monkeypatch.setattr(path_validation.os, "replace", flaky)
    with pytest.raises(PermissionError, match="final replace") as caught:
        path_validation.publish(staged)
    monkeypatch.setattr(path_validation.os, "replace", real_replace)
    contents = dict(_contents(tmp_path))
    assert contents["out0.txt"] == "OLD0"  # restored after out1 failed
    assert contents["out2.txt"] == "OLD2"
    backup = [name for name, text in contents.items() if text == "OLD1"]
    assert backup and backup[0].endswith(".bak")
    assert backup[0] in "".join(caught.value.__notes__)


def test_publish_restores_a_backup_moved_before_an_interrupt(
        tmp_path, monkeypatch):
    from deeptoolsr import path_validation
    staged = _staged(tmp_path, 2)
    real_replace = os.replace
    calls = []

    def interrupt_after_first(source, destination):
        real_replace(source, destination)
        calls.append(source)
        if len(calls) == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(path_validation.os, "replace", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        path_validation.publish(staged)
    for temporary, _ in staged:
        path_validation._discard(temporary)
    assert _contents(tmp_path) == [("out0.txt", "OLD0"), ("out1.txt", "OLD1")]


def _interrupted_publish(directory, count, existing, point):
    """Publish ``count`` outputs with an interrupt before or after the
    ``point``-th file-system call (where a real SIGINT handler can run).
    Return whether the interrupt fired."""
    from deeptoolsr import path_validation
    staged = []
    for index in range(count):
        destination = directory / f"out{index}"
        if index in existing:
            destination.write_text(f"OLD{index}")
        temporary = path_validation._reserve(str(directory), ".tmp")
        with open(temporary, "w") as handle:
            handle.write(f"NEW{index}")
        staged.append((temporary, str(destination)))
    state = {"calls": 0, "armed": True, "fired": False}

    def tick():
        if state["armed"]:
            state["calls"] += 1
            if state["calls"] == point:
                state["armed"], state["fired"] = False, True
                raise KeyboardInterrupt

    def hooked(real):
        def wrapped(*args, **kwargs):
            tick()
            result = real(*args, **kwargs)
            tick()
            return result
        return wrapped

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in ((os, "replace"), (os, "chmod"),
                            (os.path, "lexists"), (tempfile, "mkstemp")):
            patch.setattr(owner, name, hooked(getattr(owner, name)))
        try:
            path_validation.publish(staged)
        except KeyboardInterrupt:
            pass
    for temporary, _ in staged:
        path_validation._discard(temporary)
    return state["fired"]


@pytest.mark.parametrize("count, existing", [
    (1, ()), (1, (0,)), (2, ()), (2, (0, 1)), (2, (1,)),
    (3, (0, 1, 2)), (3, (0, 2)), (3, ())])
def test_publish_interrupt_never_mixes_or_loses_outputs(
        tmp_path, count, existing):
    point = 1
    while True:
        directory = tmp_path / str(point)
        directory.mkdir()
        if not _interrupted_publish(directory, count, existing, point):
            break
        old = [(f"out{i}", f"OLD{i}") for i in existing]
        new = [(f"out{i}", f"NEW{i}") for i in range(count)]
        # Stray backups are tolerated; an interrupt after the last replace
        # may leave the complete new set.
        state = [item for item in _contents(directory)
                 if not item[0].endswith(".bak")]
        assert state in (old, new), point
        point += 1
    assert point > 4 * count


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_publish_keeps_existing_permissions(tmp_path, monkeypatch):
    from deeptoolsr import path_validation
    existing = tmp_path / "private.txt"
    existing.write_text("old")
    existing.chmod(0o600)
    with atomic_output_path(existing) as temporary:
        with open(temporary, "w") as handle:
            handle.write("new")
    assert existing.stat().st_mode & 0o777 == 0o600
    fresh = tmp_path / "fresh.txt"

    def reject_chmod(*_args):
        raise PermissionError("chmod not supported")

    monkeypatch.setattr(path_validation.os, "chmod", reject_chmod)
    with atomic_output_path(fresh) as temporary:
        with open(temporary, "w") as handle:
            handle.write("new")
    assert fresh.read_text() == "new"
