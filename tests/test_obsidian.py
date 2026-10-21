import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from tools.obsidian import (
    append_obsidian_note,
    extract_tasks_summary,
    find_note_file,
    get_daily_note_path,
    get_vault_path,
    read_obsidian_note,
    sanitize_markdown_for_speech,
    search_obsidian_notes,
)
from tools.tool_registry import execute_tool


class TestObsidianIntegration(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary Obsidian vault for tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vault_path = Path(self.temp_dir.name)
        (self.vault_path / ".obsidian").mkdir(parents=True, exist_ok=True)

        # Set up realistic vault structure
        self.fee_dir = self.vault_path / "CMRU Subjects" / "FEE"
        self.fee_dir.mkdir(parents=True, exist_ok=True)
        self.fee_file = self.fee_dir / "Module 1.md"
        self.fee_file.write_text(
            """# **Diode and Applications**
```mermaid
flowchart LR
    P["P-TYPE"]
    N["N-TYPE"]
```
- **p-type:** Negative acceptor ions.
- **n-type:** Positive donor ions.
### ==🔴- Page 91 in Module 1 pdf is Definite Exam Question==
""",
            encoding="utf-8",
        )

        self.orion_dir = self.vault_path / "Projects" / "Orion"
        self.orion_dir.mkdir(parents=True, exist_ok=True)
        self.roadmap_file = self.orion_dir / "Roadmap.md"
        self.roadmap_file.write_text(
            """# Orion Roadmap
- Phase 1: Foundation and streaming STT
- Phase 2: Native tool registry and Kokoro TTS
- Phase 3: Reliability and autonomous execution
""",
            encoding="utf-8",
        )

        self.notes_file = self.vault_path / "Notes.md"
        self.notes_file.write_text(
            """# General Notes
| Topic | Status |
|---|---|
| AI Agents | Complete |
- Review [[Projects/Orion/Roadmap|Orion Roadmap]] for next steps.
""",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sanitize_markdown_for_speech(self):
        raw_markdown = """---
title: Sample Note
tags: [test, physics]
---
# Header 1
<img src="logo.png" style="width:50px"/>
Here is a [[Target Note|Custom Link Text]] and a standard [Web Link](https://example.com).
```python
def foo():
    return 42
```
| Col A | Col B |
|---|---|
| Alpha | Beta |
- **bold text** and *italic text*
- [x] Finished item
- [ ] Incomplete task
==🔴- Page 91 is Definite Exam Question==
"""
        cleaned = sanitize_markdown_for_speech(raw_markdown)

        # Asserts stripping
        self.assertNotIn("---", cleaned)
        self.assertNotIn("<img", cleaned)
        self.assertNotIn("def foo():", cleaned)
        self.assertNotIn("|---|---|", cleaned)
        self.assertNotIn("[[", cleaned)
        self.assertNotIn("]]", cleaned)
        self.assertNotIn("==", cleaned)
        self.assertIn("Custom Link Text", cleaned)
        self.assertIn("Web Link", cleaned)
        self.assertIn("bold text", cleaned)
        self.assertIn("Page 91 is Definite Exam Question", cleaned)

    def test_extract_tasks_summary(self):
        content = """# Daily Note
- [ ] Buy groceries
- [x] Attend morning sync
- [ ] Review FEE Module 1
"""
        summary = extract_tasks_summary(content)
        self.assertIsNotNone(summary)
        self.assertIn("2 pending tasks", summary)
        self.assertIn("Buy groceries", summary)
        self.assertIn("Review FEE Module 1", summary)
        self.assertIn("1 completed task", summary)

    def test_extract_tasks_summary_no_tasks(self):
        content = "Just regular prose without checkboxes."
        self.assertIsNone(extract_tasks_summary(content))

    def test_find_note_file(self):
        # 1. Exact match
        found = find_note_file(self.vault_path, "Notes")
        self.assertIsNotNone(found)
        self.assertEqual(found.stem, "Notes")

        # 2. Folder match (FEE folder -> Module 1.md)
        found_fee = find_note_file(self.vault_path, "FEE")
        self.assertIsNotNone(found_fee)
        self.assertEqual(found_fee.stem, "Module 1")

        # 3. Multi-word keywords in path (Orion Roadmap -> Projects/Orion/Roadmap.md)
        found_roadmap = find_note_file(self.vault_path, "Orion Roadmap")
        self.assertIsNotNone(found_roadmap)
        self.assertEqual(found_roadmap.stem, "Roadmap")

        # 4. Non-existent note
        missing = find_note_file(self.vault_path, "NonExistentSubjectXYZ")
        self.assertIsNone(missing)

    def test_read_obsidian_note(self):
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(self.vault_path)}):
            # Read subject note
            res = read_obsidian_note("FEE")
            self.assertIn("Module 1", res)
            self.assertIn("Page 91", res)

            # Read project note
            res_orion = read_obsidian_note("Orion Roadmap")
            self.assertIn("Roadmap", res_orion)
            self.assertIn("Reliability", res_orion)

            # Non-existent note
            res_missing = read_obsidian_note("QuantumPhysics101")
            self.assertIn("not found", res_missing.lower())

    def test_append_obsidian_note_daily_todo(self):
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(self.vault_path)}):
            today_str = datetime.now().strftime("%Y-%m-%d")
            res = append_obsidian_note("Finish AI assignment", note_name="daily", is_todo=True)
            self.assertIn("Finish AI assignment", res)

            # Check that file exists on disk with checkbox
            daily_path = self.vault_path / f"{today_str}.md"
            self.assertTrue(daily_path.is_file())
            content = daily_path.read_text(encoding="utf-8")
            self.assertIn("- [ ] Finish AI assignment", content)

            # Read back daily note
            daily_read = read_obsidian_note("daily")
            self.assertIn("1 pending task: Finish AI assignment", daily_read)

    def test_append_obsidian_note_topic_note(self):
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(self.vault_path)}):
            res = append_obsidian_note("Test new formula", note_name="FEE")
            self.assertIn("Module 1", res)

            # Verify appended with timestamp
            content = self.fee_file.read_text(encoding="utf-8")
            self.assertIn("Test new formula", content)

    def test_append_obsidian_empty_content_rejected(self):
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(self.vault_path)}):
            res = append_obsidian_note("   ")
            self.assertIn("Cannot add an empty note", res)

    def test_append_obsidian_path_traversal_blocked(self):
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(self.vault_path)}):
            res = append_obsidian_note("Malicious entry", note_name="../../evil.md")
            self.assertIn("Error", res)

    def test_search_obsidian_notes(self):
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(self.vault_path)}):
            # Keyword in content
            res_content = search_obsidian_notes("Definite Exam Question")
            self.assertIn("Module 1", res_content)

            # Keyword in path/title
            res_path = search_obsidian_notes("Orion Roadmap")
            self.assertIn("Roadmap", res_path)

            # Missing query
            res_none = search_obsidian_notes("ArbitraryWordNotInAnyFile")
            self.assertIn("No Obsidian notes found", res_none)

    def test_tool_registry_execution(self):
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(self.vault_path)}):
            # 1. Execute search
            search_res = execute_tool("search_obsidian_notes", {"query": "Roadmap"})
            self.assertIn("Roadmap", search_res)

            # 2. Execute read
            read_res = execute_tool("read_obsidian_note", {"note_name": "FEE"})
            self.assertIn("Page 91", read_res)

            # 3. Execute append
            append_res = execute_tool(
                "append_obsidian_note",
                {"content": "Prepare for viva", "note_name": "To-Do", "is_todo": True},
            )
            self.assertIn("Prepare for viva", append_res)


if __name__ == "__main__":
    unittest.main()
