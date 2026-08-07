import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import tempfile

from modules.repo_ingestion import ingest_repository, parse_gitignore, is_ignored_by_gitignore, contains_secret
from modules.context_builder import build_context, _extract_keywords
from modules.code_modifier import CodeModificationEngine, _apply_unified_diff, FileChange
from modules.sandbox import SubprocessSandbox, ExecutionResult
from modules.llm_client import LLMClient, BaseLLMClient


class TestRepoIngestion(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_gitignore_parsing(self):
        with open(os.path.join(self.test_dir, ".gitignore"), "w") as f:
            f.write("*.log\n# comment\n/build/\n")
        patterns = parse_gitignore(self.test_dir)
        self.assertIn("*.log", patterns)
        self.assertIn("/build/", patterns)
        
    def test_gitignore_ignoring(self):
        patterns = ["*.log", "build/"]
        self.assertTrue(is_ignored_by_gitignore("error.log", patterns))
        self.assertTrue(is_ignored_by_gitignore("build/output.txt", patterns))
        self.assertFalse(is_ignored_by_gitignore("src/main.py", patterns))
        
    def test_secret_detection(self):
        self.assertTrue(contains_secret("sk-ant-a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0"))
        self.assertTrue(contains_secret("sk-1234567890abcdef1234567890abcdef1234567890abcdef"))
        self.assertFalse(contains_secret("regular content without keys"))


class TestContextBuilder(unittest.TestCase):
    def test_extract_keywords(self):
        keywords = _extract_keywords("Fix the TypeError in the parse function")
        self.assertIn("typeerror", keywords)
        self.assertIn("parse", keywords)
        self.assertIn("fix", keywords)


class TestCodeModifier(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.engine = CodeModificationEngine(self.test_dir, os.path.join(self.test_dir, "backups"))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_apply_patch(self):
        original = "line 1\nline 2\nline 3\n"
        patch_text = "@@ -2,2 +2,2 @@\n-line 2\n+line 2 modified\n line 3\n"
        result = _apply_unified_diff(original, patch_text)
        self.assertEqual(result, "line 1\nline 2 modified\nline 3\n")

    def test_modify_file(self):
        file_path = os.path.join(self.test_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("hello")
            
        change = FileChange(path="test.txt", action="modify", content="world", explanation="test")
        res = self.engine.apply_changes([change])
        self.assertTrue(res[0].success)
        
        with open(file_path, "r") as f:
            self.assertEqual(f.read(), "world")


class TestLLMClient(unittest.TestCase):
    def test_estimate_tokens(self):
        client = BaseLLMClient()
        text = "a" * 40
        self.assertEqual(client._estimate_tokens(text), 10)


if __name__ == "__main__":
    unittest.main()
