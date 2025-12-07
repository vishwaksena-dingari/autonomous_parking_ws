#!/usr/bin/env python3
"""
Dump an entire project directory into a single Markdown file.

Features:
- Skips hidden files/directories (names starting with '.')
- Skips files whose names contain 'bak' (case-insensitive)
- Skips 'results', 'images', '__pycache__', 'old', and 'archive' directories (case-insensitive)
- Skips image files (.png, .jpg, etc.), .log, and .mp4 files
- Skips obvious binary files
- Markdown structure:
    # Project Dump
    Project path: `...`

    ## Project Tree
    ```text
    <tree>
    ```

    ## Files
    ### path/to/file.ext
    ```<language-or-extension>
    <file contents>
    ```
"""

import os
import argparse
from pathlib import Path

# File types to skip entirely in "Files" section
IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".svg",
    ".ico",
}
SKIP_FILE_EXTS = {".md", ".log", ".mp4", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".svg", ".ico", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar"}

# Directories to ignore (case-insensitive)
SKIP_DIR_NAMES = {"logs", "results", "images", "__pycache__", "old", "archive", "_scratch"}


def is_hidden(path: Path) -> bool:
    """Return True if any part of the path starts with '.'."""
    return any(part.startswith(".") for part in path.parts)


def has_bak_in_name(path: Path) -> bool:
    """Return True if filename contains 'bak' (case-insensitive)."""
    return "bak" in path.name.lower()


def is_image_file(path: Path) -> bool:
    """Return True if file extension is an image extension."""
    return path.suffix.lower() in IMAGE_EXTS


def is_skipped_by_ext(path: Path) -> bool:
    """Return True if file should be skipped based on extension."""
    ext = path.suffix.lower()
    return ext in SKIP_FILE_EXTS or ext in IMAGE_EXTS


def is_binary_file(path: Path, chunk_size: int = 1024) -> bool:
    """Rudimentary binary file detection."""
    try:
        with path.open("rb") as f:
            chunk = f.read(chunk_size)
        if b"\0" in chunk:
            return True
    except Exception:
        return True
    return False


def language_from_extension(ext: str) -> str:
    """Return markdown language tag.

    - If extension is known: return mapped language.
    - If unknown: return extension without dot.
    - If no extension: return empty string.
    """
    ext = ext.lower()

    mapping = {
        ".py": "python",
        ".ipynb": "json",
        ".md": "markdown",
        ".txt": "text",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".ps1": "powershell",
        ".bat": "bat",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".java": "java",
        ".kt": "kotlin",
        ".rs": "rust",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".sql": "sql",
        ".r": "r",
        ".jl": "julia",
        ".xml": "xml",
        ".ini": "ini",
        ".cfg": "ini",
        ".toml": "toml",
        ".csv": "csv",
        ".tsv": "tsv",
    }

    if ext in mapping:
        return mapping[ext]

    if ext.startswith(".") and len(ext) > 1:
        return ext[1:]

    return ""


def generate_tree(root: Path, ignore_paths) -> str:
    """Generate a tree structure."""
    lines = []

    def walk(dir_path: Path, prefix: str = ""):
        entries = []
        for child in sorted(dir_path.iterdir(), key=lambda p: p.name.lower()):
            if child in ignore_paths:
                continue
            if is_hidden(child):
                continue
            if has_bak_in_name(child):
                continue
            # Skip certain directories
            if child.is_dir():
                if child.name.lower() in SKIP_DIR_NAMES:
                    continue
            else:
                # Skip files by extension (logs, images, videos, archives, etc.)
                if is_skipped_by_ext(child):
                    continue

            entries.append(child)

        for i, child in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            rel_child = child.relative_to(root)
            lines.append(f"{prefix}{connector}{rel_child.name}")
            if child.is_dir():
                ext_prefix = "    " if i == len(entries) - 1 else "│   "
                walk(child, prefix + ext_prefix)

    lines.append(root.name)
    walk(root)
    return "\n".join(lines)


def collect_files(root: Path, output_file: Path):
    """Collect all non-ignored files."""
    files = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)

        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d.lower() not in SKIP_DIR_NAMES
            and "bak" not in d.lower()
        ]

        if is_hidden(dirpath):
            continue

        for filename in sorted(filenames, key=str.lower):
            file_path = dirpath / filename

            if file_path == output_file:
                continue
            if is_hidden(file_path):
                continue
            if has_bak_in_name(file_path):
                continue
            if is_skipped_by_ext(file_path):
                continue

            files.append(file_path)

    return sorted(files, key=lambda p: str(p).lower())


def read_text_file(path: Path) -> str:
    """Read file as text."""
    if is_binary_file(path):
        return None

    for encoding in ("utf-8", "latin-1"):
        try:
            with path.open("r", encoding=encoding) as f:
                return f.read()
        except Exception:
            continue

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path", help="Path to project.")
    parser.add_argument("-o", "--output", default="project_dump.md")

    args = parser.parse_args()

    root = Path(args.project_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Error: {root} is not a directory.")

    ignore_paths = {output_path}

    tree_str = generate_tree(root, ignore_paths)
    files = collect_files(root, output_path)

    with output_path.open("w", encoding="utf-8") as out:
        out.write("# Project Dump\n\n")
        out.write(f"Project path: `{root}`\n\n")

        out.write("## Project Tree\n\n```text\n")
        out.write(tree_str + "\n```\n\n")

        out.write("## Files\n\n")

        for file_path in files:
            rel = file_path.relative_to(root)
            out.write(f"### `{rel}`\n\n")

            content = read_text_file(file_path)
            if content is None:
                out.write("_Skipped (binary or unreadable file)_\n\n")
                continue

            lang = language_from_extension(file_path.suffix)

            if lang:
                out.write(f"```{lang}\n")
            else:
                out.write("```\n")

            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            out.write("```\n\n")

    print(f"Markdown dump written to: {output_path}")


if __name__ == "__main__":
    main()
