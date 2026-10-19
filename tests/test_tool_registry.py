import unittest
from unittest.mock import patch

from tools.tool_registry import (
    AVAILABLE_TOOL_SCHEMAS,
    TOOLS,
    execute_tool,
)


class TestToolRegistry(unittest.TestCase):
    def test_tool_schemas_integrity(self):
        # 1. Ensure schemas list is non-empty
        self.assertGreater(len(AVAILABLE_TOOL_SCHEMAS), 0)

        # 2. Verify all tools in schemas exist in TOOLS mapping
        schema_tool_names = set()
        for item in AVAILABLE_TOOL_SCHEMAS:
            self.assertEqual(item.get("type"), "function")
            fn = item.get("function", {})
            name = fn.get("name")
            self.assertIsNotNone(name)
            self.assertIn(name, TOOLS)
            schema_tool_names.add(name)

            # Ensure valid parameters structure
            params = fn.get("parameters", {})
            self.assertEqual(params.get("type"), "object")
            self.assertIn("properties", params)

        # 3. Ensure all TOOLS have a corresponding schema definition
        for tool_name in TOOLS:
            self.assertIn(tool_name, schema_tool_names)

    def test_execute_tool_unknown_name(self):
        result = execute_tool("nonexistent_utility_xyz", {"param": 123})
        self.assertIn("Error: Unknown tool", result)
        self.assertIn("nonexistent_utility_xyz", result)

    def test_execute_tool_web_search_empty_query(self):
        result = execute_tool("web_search", {})
        self.assertIn("Error: No search query specified", result)

        result_empty = execute_tool("web_search", {"query": "   "})
        self.assertIn("Error: No search query specified", result_empty)

    def test_execute_tool_argument_filtering(self):
        # Tools without **kwargs (like list_gnome_alarms or list_reminders) should discard extra args
        with patch.dict(TOOLS, {"mock_tool": lambda a: f"result_{a}"}):
            res = execute_tool("mock_tool", {"a": "valid", "extra_bogus_arg": 999})
            self.assertEqual(res, "result_valid")

    def test_execute_tool_exception_handling(self):
        def faulty_tool():
            raise RuntimeError("Database connection crashed")

        with patch.dict(TOOLS, {"faulty_tool": faulty_tool}):
            res = execute_tool("faulty_tool", {})
            self.assertIn("Tool execution error", res)
            self.assertIn("Database connection crashed", res)


if __name__ == "__main__":
    unittest.main()
