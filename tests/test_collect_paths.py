import os

from mgds.pipelineModules.CollectPaths import CollectPaths

import pytest


def _collector() -> CollectPaths:
    return CollectPaths(
        concept_in_name="concept",
        path_in_name="path",
        include_subdirectories_in_name="concept.include_subdirectories",
        enabled_in_name="concept.enabled",
        path_out_name="image_path",
        concept_out_name="concept",
        extensions=[".png", ".jpg"],
        include_postfix=None,
        exclude_postfix=[],
    )


def _list_files(path, include_subdirectories=True):
    collector = _collector()
    return collector._CollectPaths__list_files(str(path), include_subdirectories)


def test_list_files_recurses_without_hidden_directories(tmp_path):
    root = tmp_path / "root"
    nested = root / "nested"
    hidden = root / ".hidden"
    nested.mkdir(parents=True)
    hidden.mkdir()

    root_file = root / "root.png"
    nested_file = nested / "nested.jpg"
    hidden_file = hidden / "hidden.jpg"
    root_file.write_bytes(b"root")
    nested_file.write_bytes(b"nested")
    hidden_file.write_bytes(b"hidden")

    files = {os.path.relpath(path, root) for path in _list_files(root)}

    assert files == {"root.png", os.path.join("nested", "nested.jpg")}


def test_list_files_respects_non_recursive_mode(tmp_path):
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)

    root_file = root / "root.png"
    nested_file = nested / "nested.jpg"
    root_file.write_bytes(b"root")
    nested_file.write_bytes(b"nested")

    files = {os.path.relpath(path, root) for path in _list_files(root, include_subdirectories=False)}

    assert files == {"root.png"}


def test_list_files_does_not_follow_symlink_cycles(tmp_path):
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    nested_file = nested / "nested.jpg"
    nested_file.write_bytes(b"nested")

    link = nested / "loop"
    try:
        link.symlink_to(root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    files = [os.path.relpath(path, root) for path in _list_files(root)]

    assert files == [os.path.join("nested", "nested.jpg")]
