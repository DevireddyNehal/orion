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

STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "my", "your", "under", "folder", "directory", "file", "note", "notes",
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

    for d in base.iterdir():
        if d.is_dir() and not d.name.startswith(".") and (d / ".obsidian").is_dir():
            return d

    return base


def get_daily_note_path(vault_path: Path, date_str: str | None = None) -> Path:
    """
    Resolves the file path for a daily note (YYYY-MM-DD.md).
    Checks OBSIDIAN_DAILY_DIR env var, common subfolders, or defaults to vault root.
    """
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    filename = f"{date_str}.md"
    env_dir = os.getenv("OBSIDIAN_DAILY_DIR")
    if env_dir:
        return vault_path / env_dir / filename

    for folder_name in ["Daily", "Daily Notes", "daily", "Journal", "journal"]:
        candidate = vault_path / folder_name
        if candidate.is_dir():
            return candidate / filename

    return vault_path / filename


def sanitize_markdown_for_speech(text: str, max_chars: int = 4000) -> str:
    """
    Strips YAML frontmatter, markdown links, code blocks, headers, bullet formatting,
    highlight markers, and table borders to produce clean, readable text.
    """
    if not text:
        return ""

    # 1. Strip YAML frontmatter at start of file
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)

    # 2. Strip HTML tags like <img>, <div>, etc.
    text = re.sub(r"<[^>]+>", "", text)

    # 3. Strip fenced code blocks (3 or more backticks)
    text = re.sub(r"`{3,}[\s\S]*?`{3,}", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"`+", "", text)

    # 4. Strip Obsidian highlights ==text== -> text
    text = re.sub(r"==([^=]+)==", r"\1", text)

    # 5. Strip image embeds ![[...]] and ![alt](url)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # 6. Strip markdown table separator lines and replace pipes with space
    text = re.sub(r"\|[\s\-:|]+\|", " ", text)
    text = text.replace("|", " ")

    # 7. Strip Obsidian wikilinks [[Path/To/Page|Display Text]] -> Display Text, [[Page]] -> Page
    def _replace_wikilink(match: re.Match[str]) -> str:
        inner = match.group(1)
        if "|" in inner:
            return inner.split("|", 1)[1].strip()
        return inner.split("/")[-1].strip()

    text = re.sub(r"\[\[(.*?)\]\]", _replace_wikilink, text)

    # 8. Strip standard markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # 9. Strip Markdown headers, blockquotes, horizontal rules, and list markers
    text = re.sub(r"^[#\s*>-]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s*(\[[ xX]\]\s*)?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)

    # 10. Strip bold and italics (*, _, **, __)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)

    # 11. Collapse whitespace and empty lines into natural sentence separators
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 12. Truncate cleanly to ~max_chars if necessary
    if len(text) > max_chars:
        truncated = text[:max_chars]
        last_period = truncated.rfind(". ")
        if last_period > int(max_chars * 0.7):
            text = truncated[: last_period + 1]
        else:
            last_space = truncated.rfind(" ")
            if last_space > 0:
                text = truncated[:last_space] + "..."
            else:
                text = truncated + "..."

    return text.strip()


def extract_tasks_summary(raw_text: str) -> str | None:
    """
    Extracts pending and completed tasks from note content for voice output.
    """
    pending: list[str] = []
    completed: list[str] = []

    for line in raw_text.splitlines():
        line_clean = line.strip()
        if re.match(r"^[-*+]\s+\[ \]\s+", line_clean):
            task_text = re.sub(r"^[-*+]\s+\[ \]\s+", "", line_clean).strip()
            task_clean = sanitize_markdown_for_speech(task_text, max_chars=80)
            if task_clean:
                pending.append(task_clean)
        elif re.match(r"^[-*+]\s+\[[xX]\]\s+", line_clean):
            task_text = re.sub(r"^[-*+]\s+\[[xX]\]\s+", "", line_clean).strip()
            task_clean = sanitize_markdown_for_speech(task_text, max_chars=80)
            if task_clean:
                completed.append(task_clean)

    if not pending and not completed:
        return None

    parts: list[str] = []
    if pending:
        task_str = ", ".join(pending)
        parts.append(f"{len(pending)} pending task{'s' if len(pending) > 1 else ''}: {task_str}")
    if completed:
        parts.append(f"{len(completed)} completed task{'s' if len(completed) > 1 else ''}")

    return "; ".join(parts)


def _is_safe_path(base: Path, target: Path) -> bool:
    """Prevents directory traversal outside the vault."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def find_note_file(vault_path: Path, note_name: str) -> Path | None:
    """
    Locates a note file in the vault by exact path, filename, folder name, or keyword matching across path.
    Searches projects, subjects, roadmaps, and documents.
    """
    clean_name = note_name.strip()
    if not clean_name:
        return None

    clean_stem = clean_name.removesuffix(".md").strip()
    target_clean = clean_stem.lower()

    # 1. Direct relative path check
    direct_file = (vault_path / f"{clean_stem}.md").resolve()
    if _is_safe_path(vault_path, direct_file) and direct_file.is_file():
        return direct_file

    direct_exact = (vault_path / clean_name).resolve()
    if _is_safe_path(vault_path, direct_exact) and direct_exact.is_file():
        return direct_exact

    # Extract meaningful keywords
    raw_words = re.findall(r"\w+", target_clean)
    keywords = [w for w in raw_words if w not in STOP_WORDS]
    if not keywords:
        keywords = raw_words

    exact_file_match: Path | None = None
    all_keywords_match: Path | None = None
    folder_match_file: Path | None = None
    partial_match: Path | None = None

    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

        folder_name = Path(root).name.lower()
        if (folder_name == target_clean or target_clean in folder_name) and folder_match_file is None:
            md_files = [Path(root) / f for f in files if f.lower().endswith(".md")]
            if md_files:
                folder_match_file = md_files[0]

        for file in files:
            if not file.lower().endswith(".md"):
                continue

            file_stem = Path(file).stem.lower()
            rel_path = (Path(root) / file).relative_to(vault_path).as_posix().lower()

            if file_stem == target_clean:
                exact_file_match = Path(root) / file
                return exact_file_match

            # Match all keywords in relative path (e.g. 'Orion' and 'Roadmap' in 'Projects/Orion/Roadmap.md')
            if keywords and all(k in rel_path for k in keywords):
                all_keywords_match = Path(root) / file
                return all_keywords_match

            if target_clean in file_stem and partial_match is None:
                partial_match = Path(root) / file

    return all_keywords_match or folder_match_file or partial_match


def append_obsidian_note(
    content: str,
    note_name: str | None = None,
    is_todo: bool = False,
) -> str:
    """
    Appends a note or to-do task to an Obsidian note (projects, subjects, roadmaps, or daily to-dos).
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

    clean_target = (note_name or "").strip()
    target_lower = clean_target.lower()

    is_daily = target_lower in ("daily", "today", "daily note", "daily todo", "daily to-do", "daily tasks")

    if is_daily or any(k in target_lower for k in ["todo", "to-do", "task"]):
        is_todo = True

    if is_daily:
        today_str = datetime.now().strftime("%Y-%m-%d")
        target_path = get_daily_note_path(vault_path, today_str)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        display_title = f"today's daily note ({today_str})"
    elif not clean_target:
        if is_todo:
            today_str = datetime.now().strftime("%Y-%m-%d")
            today_candidate = get_daily_note_path(vault_path, today_str)
            if today_candidate.is_file():
                target_path = today_candidate
                display_title = f"today's daily note ({today_str})"
                is_daily = True
            else:
                existing_todo = find_note_file(vault_path, "To-Do") or find_note_file(vault_path, "Tasks")
                if existing_todo:
                    target_path = existing_todo
                    display_title = target_path.stem
                else:
                    target_path = get_daily_note_path(vault_path, today_str)
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    display_title = f"today's daily note ({today_str})"
                    is_daily = True
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

    file_exists = target_path.is_file()
    file_empty = not file_exists or target_path.stat().st_size == 0

    if is_todo:
        entry_line = f"- [ ] {note_text}\n"
    else:
        timestamp = datetime.now().strftime("%H:%M")
        entry_line = f"- [{timestamp}] {note_text}\n"

    try:
        with open(target_path, "a", encoding="utf-8") as f:
            if file_empty:
                if is_daily:
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    f.write(f"# {today_str}\n\n## Tasks\n")
                else:
                    f.write(f"# {target_path.stem}\n\n")
            f.write(entry_line)

        logger.info(f"[Obsidian] Appended to {target_path}: {entry_line.strip()}")
        action_desc = "task to your to-do list" if is_todo else f"note to {display_title}"
        if file_empty:
            return f"Created {display_title} and added {action_desc}: '{note_text}'."
        return f"Added {action_desc}: '{note_text}'."
    except Exception as e:
        logger.exception(f"[Obsidian] Error writing to note {target_path}: {e}")
        return f"Error saving to Obsidian: {e}"


def read_obsidian_note(note_name: str = "", max_chars: int = 4000) -> str:
    """
    Reads and summarizes an Obsidian note, project roadmap, meeting log, or to-do tasks.

    Args:
        note_name: The title, subject, project, or date of the note to read
                   (e.g., 'Orion Roadmap', 'Projects/Orion', 'daily', 'Module 1', 'FEE', 'To-Do', 'Notes').
                   If 'daily' or empty, reads today's daily note or general notes.
        max_chars: Maximum characters to return (default: 4000).

    Returns:
        The content or pending tasks of the note.
    """
    vault_path = get_vault_path()
    if not vault_path.exists():
        return "Obsidian vault path does not exist."

    clean_name = (note_name or "").strip()
    is_daily = clean_name.lower() in ("daily", "today", "daily note", "daily todo", "daily to-do", "daily tasks")

    if is_daily:
        today_str = datetime.now().strftime("%Y-%m-%d")
        target_path = get_daily_note_path(vault_path, today_str)
        if not target_path.is_file():
            root_fallback = vault_path / f"{today_str}.md"
            if root_fallback.is_file():
                target_path = root_fallback
        display_title = f"today's daily note ({today_str})"
    elif not clean_name:
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily_candidate = get_daily_note_path(vault_path, today_str)
        if daily_candidate.is_file():
            target_path = daily_candidate
            display_title = f"today's note ({today_str})"
        else:
            target_path = find_note_file(vault_path, "Notes") or find_note_file(vault_path, "To-Do")
            if not target_path:
                return "No default 'Notes' or daily note found in your vault."
            display_title = target_path.stem
    else:
        target_path = find_note_file(vault_path, clean_name)
        if not target_path or not target_path.is_file():
            return f"Note '{clean_name}' was not found in your Obsidian vault."
        display_title = target_path.stem

    if not target_path.is_file():
        return f"Note '{display_title}' does not exist yet."

    try:
        raw_text = target_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return f"The note {display_title} is empty."

        tasks_summary = extract_tasks_summary(raw_text)
        clean_text = sanitize_markdown_for_speech(raw_text, max_chars=max_chars)

        if tasks_summary and is_daily:
            return f"In {display_title}: {tasks_summary}."

        if not clean_text:
            return f"The note {display_title} has no readable text."

        return f"{display_title}: {clean_text}"
    except Exception as e:
        logger.exception(f"[Obsidian] Error reading note {target_path}: {e}")
        return f"Error reading Obsidian note: {e}"


def search_obsidian_notes(query: str, max_results: int = 5) -> str:
    """
    Searches Obsidian notes by keyword across projects, roadmaps, subjects, and to-do lists.

    Args:
        query: The search term or keywords (e.g. 'Orion Roadmap', 'FEE', 'Projects/Orion', 'exam').
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

    raw_words = re.findall(r"\w+", query_term)
    keywords = [w for w in raw_words if w not in STOP_WORDS]
    if not keywords:
        keywords = raw_words

    title_matches: list[tuple[str, str]] = []
    content_matches: list[tuple[str, str]] = []

    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        folder_name = Path(root).name.lower()

        for file in files:
            if not file.lower().endswith(".md"):
                continue

            file_path = Path(root) / file
            stem = file_path.stem
            stem_lower = stem.lower()
            rel_path = file_path.relative_to(vault_path).as_posix().lower()

            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            clean_content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
            clean_snippet = sanitize_markdown_for_speech(clean_content, max_chars=180)

            # Match title, relative path, or keywords
            path_matches_all = keywords and all(k in rel_path for k in keywords)
            if query_term in stem_lower or query_term in rel_path or path_matches_all:
                title_matches.append((stem, clean_snippet or "No additional content"))
            elif query_term in content.lower() or (keywords and all(k in content.lower() for k in keywords)):
                lower_c = clean_content.lower()
                first_k = keywords[0] if keywords else query_term
                idx = lower_c.find(first_k)
                if idx != -1:
                    start = max(0, idx - 40)
                    end = min(len(clean_content), idx + len(first_k) + 120)
                    snippet = clean_content[start:end].replace("\n", " ").strip()
                    snippet = sanitize_markdown_for_speech(snippet, max_chars=180)
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
