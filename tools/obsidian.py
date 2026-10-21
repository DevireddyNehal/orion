import os
import re
from datetime import datetime
from pathlib import Path

from loguru import logger

# Directories to ignore when traversing Obsidian vault
IGNORED_DIRS = {
    ".obsidian",
    ".trash",
    ".git",
    ".smart-env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}


def get_vault_path() -> Path:
    """
    Discovers the Obsidian vault path.
    1. Checks OBSIDIAN_VAULT_PATH environment variable.
    2. Checks ~/Obsidian/Vaults subfolder for active vaults.
    3. Checks ~/Obsidian and direct subfolders for .obsidian.
    4. Defaults to ~/Obsidian if none found.
    """
    env_path = os.getenv("OBSIDIAN_VAULT_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    base = Path.home() / "Obsidian"
    if not base.exists():
        return base

    # Check ~/Obsidian/Vaults subfolder (e.g., 'Laptop Vault')
    vaults_dir = base / "Vaults"
    if vaults_dir.is_dir():
        subdirs = [
            d for d in vaults_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        for d in subdirs:
            if (d / ".obsidian").is_dir():
                return d
        if subdirs:
            return subdirs[0]

    # Check direct subdirectories of ~/Obsidian
    for d in base.iterdir():
        if d.is_dir() and not d.name.startswith(".") and (d / ".obsidian").is_dir():
            return d

    return base


def sanitize_markdown_for_speech(text: str, max_chars: int = 300) -> str:
    """
    Strips YAML frontmatter, markdown links, code blocks, headers, bullet formatting,
    and markdown table borders to produce clean, speech-friendly text for Kokoro TTS.
    """
    if not text:
        return ""

    # 1. Strip YAML frontmatter at start of file
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)

    # 2. Strip HTML tags like <img>, <div>, etc.
    text = re.sub(r"<[^>]+>", "", text)

    # 3. Strip code blocks and inline backticks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # 3. Strip image embeds ![[...]] and ![alt](url)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # 4. Strip markdown table separator lines and replace pipes with space
    text = re.sub(r"\|[\s\-:|]+\|", " ", text)
    text = text.replace("|", " ")

    # 5. Strip Obsidian wikilinks [[Path/To/Page|Display Text]] -> Display Text, [[Page]] -> Page
    def _replace_wikilink(match: re.Match[str]) -> str:
        inner = match.group(1)
        if "|" in inner:
            return inner.split("|", 1)[1].strip()
        return inner.split("/")[-1].strip()

    text = re.sub(r"\[\[(.*?)\]\]", _replace_wikilink, text)

    # 6. Strip standard markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # 7. Strip Markdown headers, blockquotes, horizontal rules, and list markers
    text = re.sub(r"^[#\s*>-]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s*(\[[ xX]\]\s*)?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)

    # 8. Strip bold and italics (*, _, **, __)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)

    # 9. Collapse whitespace and empty lines
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 10. Truncate cleanly to ~max_chars at sentence or word boundary
    if len(text) > max_chars:
        truncated = text[:max_chars]
        last_period = truncated.rfind(". ")
        if last_period > int(max_chars * 0.5):
            text = truncated[: last_period + 1]
        else:
            last_space = truncated.rfind(" ")
            if last_space > 0:
                text = truncated[:last_space] + "..."
            else:
                text = truncated + "..."

    return text.strip()


def _is_safe_path(base: Path, target: Path) -> bool:
    """Prevents directory traversal outside the vault."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def find_note_file(vault_path: Path, note_name: str) -> Path | None:
    """
    Locates a note file in the vault by exact path, filename, or substring matching.
    Searches subject folders (e.g., CMRU Subjects, Projects, Roadmap).
    """
    clean_name = note_name.strip()
    if not clean_name:
        return None

    if not clean_name.lower().endswith(".md"):
        clean_name += ".md"

    # 1. Direct relative path check
    direct_target = (vault_path / clean_name).resolve()
    if _is_safe_path(vault_path, direct_target) and direct_target.is_file():
        return direct_target

    target_filename = Path(clean_name).name.lower()
    target_stem = Path(clean_name).stem.lower()

    exact_match: Path | None = None
    partial_match: Path | None = None

    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        for file in files:
            if not file.lower().endswith(".md"):
                continue

            file_stem = Path(file).stem.lower()
            if file.lower() == target_filename or file_stem == target_stem:
                exact_match = Path(root) / file
                return exact_match

            if target_stem in file_stem and partial_match is None:
                partial_match = Path(root) / file

    return partial_match


def append_obsidian_note(
    content: str,
    note_name: str | None = None,
    is_todo: bool = False,
) -> str:
    """
    Appends a note or to-do task to an Obsidian note (subject notes, to-do lists, or general notes).

    Args:
        content: The text content of the note or task.
        note_name: Subject, project, or note title (e.g. 'FEE', 'Projects/Orion', 'To-Do', 'Notes').
                   If None or empty, defaults to 'To-Do' for tasks or 'Notes.md'.
        is_todo: If True, formats entry as an Obsidian checkbox task: '- [ ] content'.
                 Otherwise appends as a timestamped note: '- [HH:MM] content'.

    Returns:
        Speech-friendly confirmation message.
    """
    if not content or not content.strip():
        return "Cannot add an empty note."

    note_text = content.strip()
    vault_path = get_vault_path()

    try:
        vault_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"[Obsidian] Failed to access vault at {vault_path}: {e}")
        return f"Error accessing Obsidian vault: {e}"

    # Auto-detect to-do intent if note_name mentions todo or task
    clean_target = (note_name or "").strip()
    target_lower = clean_target.lower()
    if any(k in target_lower for k in ["todo", "to-do", "task"]):
        is_todo = True

    # Default note resolution:
    if not clean_target:
        if is_todo:
            existing_todo = find_note_file(vault_path, "To-Do") or find_note_file(vault_path, "Tasks")
            if existing_todo:
                target_path = existing_todo
                display_title = target_path.stem
            else:
                target_path = vault_path / "To-Do.md"
                display_title = "To-Do list"
        else:
            existing_notes = find_note_file(vault_path, "Notes")
            if existing_notes:
                target_path = existing_notes
                display_title = target_path.stem
            else:
                target_path = vault_path / "Notes.md"
                display_title = "Notes"
    else:
        existing_file = find_note_file(vault_path, clean_target)
        if existing_file:
            target_path = existing_file
            display_title = target_path.stem
        else:
            filename = clean_target if clean_target.lower().endswith(".md") else f"{clean_target}.md"
            target_path = (vault_path / filename).resolve()
            if not _is_safe_path(vault_path, target_path):
                return "Error: Invalid note path."
            target_path.parent.mkdir(parents=True, exist_ok=True)
            display_title = target_path.stem

    # Formatting line: checkbox for to-do vs timestamped entry for notes
    if is_todo:
        entry_line = f"- [ ] {note_text}\n"
    else:
        timestamp = datetime.now().strftime("%H:%M")
        entry_line = f"- [{timestamp}] {note_text}\n"

    created = not target_path.exists()
    try:
        with open(target_path, "a", encoding="utf-8") as f:
            if created:
                f.write(f"# {target_path.stem}\n\n")
            f.write(entry_line)

        logger.info(f"[Obsidian] Appended to {target_path}: {entry_line.strip()}")
        action_desc = "task to your to-do list" if is_todo else f"note to {display_title}"
        if created:
            return f"Created {display_title} and added {action_desc}: '{note_text}'."
        return f"Added {action_desc}: '{note_text}'."
    except Exception as e:
        logger.exception(f"[Obsidian] Error writing to note {target_path}: {e}")
        return f"Error saving to Obsidian: {e}"


def read_obsidian_note(note_name: str = "") -> str:
    """
    Reads and summarizes an Obsidian note.

    Args:
        note_name: The title, subject, or relative path of the note
                   (e.g., 'FEE Module 1', 'Notes', 'Projects/Orion', 'To-Do').
                   If empty, defaults to 'Notes' or 'To-Do'.

    Returns:
        Speech-friendly summary of the note's content.
    """
    vault_path = get_vault_path()
    if not vault_path.exists():
        return "Obsidian vault path does not exist."

    clean_name = (note_name or "").strip()
    if not clean_name:
        target_path = find_note_file(vault_path, "Notes") or find_note_file(vault_path, "To-Do")
        if not target_path:
            return "No default 'Notes' or 'To-Do' file found in your vault."
        display_title = target_path.stem
    else:
        target_path = find_note_file(vault_path, clean_name)
        if not target_path or not target_path.is_file():
            return f"Note '{clean_name}' was not found in your Obsidian vault."
        display_title = target_path.stem

    try:
        raw_text = target_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return f"The note {display_title} is empty."

        clean_text = sanitize_markdown_for_speech(raw_text)
        if not clean_text:
            return f"The note {display_title} has no readable text."

        return f"{display_title}: {clean_text}"
    except Exception as e:
        logger.exception(f"[Obsidian] Error reading note {target_path}: {e}")
        return f"Error reading Obsidian note: {e}"


def search_obsidian_notes(query: str, max_results: int = 5) -> str:
    """
    Searches Obsidian notes by keyword across subjects, projects, and to-do lists.

    Args:
        query: The search term or keywords (e.g. 'homework', 'FEE', 'exam', 'Orion').
        max_results: Maximum matching notes to return (default: 5).

    Returns:
        Speech-formatted list of matching notes and snippets.
    """
    if not query or not query.strip():
        return "Please specify a search query for your Obsidian vault."

    vault_path = get_vault_path()
    if not vault_path.exists():
        return "Obsidian vault path does not exist."

    query_term = query.strip().lower()
    max_results = max(1, min(max_results, 10))

    title_matches: list[tuple[str, str]] = []
    content_matches: list[tuple[str, str]] = []

    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        for file in files:
            if not file.lower().endswith(".md"):
                continue

            file_path = Path(root) / file
            stem = file_path.stem
            stem_lower = stem.lower()

            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            clean_content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
            clean_snippet = sanitize_markdown_for_speech(clean_content, max_chars=120)

            if query_term in stem_lower:
                title_matches.append((stem, clean_snippet or "No additional content"))
            elif query_term in content.lower():
                lower_c = clean_content.lower()
                idx = lower_c.find(query_term)
                if idx != -1:
                    start = max(0, idx - 40)
                    end = min(len(clean_content), idx + len(query_term) + 80)
                    snippet = clean_content[start:end].replace("\n", " ").strip()
                    snippet = sanitize_markdown_for_speech(snippet, max_chars=120)
                else:
                    snippet = clean_snippet
                content_matches.append((stem, snippet or "Matched content"))

    results = title_matches + content_matches
    if not results:
        return f"No Obsidian notes found matching '{query}'."

    unique_results: list[tuple[str, str]] = []
    seen_titles: set[str] = set()
    for title, snip in results:
        if title not in seen_titles:
            seen_titles.add(title)
            unique_results.append((title, snip))
            if len(unique_results) >= max_results:
                break

    summaries = [f"{t}: {s}" for t, s in unique_results]
    return f"Found {len(unique_results)} matching note{'s' if len(unique_results) != 1 else ''}: " + "; ".join(summaries)
