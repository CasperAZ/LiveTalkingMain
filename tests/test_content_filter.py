import tempfile
import time
import unittest
from pathlib import Path

from content_filter import SensitiveWordFilter


class SensitiveWordFilterTest(unittest.TestCase):
    def test_loads_words_and_blocks_matching_text(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sensitive_words.txt"
            path.write_text("# comment\n测试违禁词\n\nBlockedWord\n", encoding="utf-8")

            word_filter = SensitiveWordFilter(str(path))

            self.assertTrue(word_filter.check("这里有测试违禁词").blocked)
            self.assertTrue(word_filter.check("contains blockedword").blocked)
            self.assertFalse(word_filter.check("正常直播互动").blocked)

    def test_reload_when_file_changes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "sensitive_words.txt"
            path.write_text("旧词\n", encoding="utf-8")

            word_filter = SensitiveWordFilter(str(path))
            self.assertTrue(word_filter.check("旧词").blocked)

            time.sleep(0.02)
            path.write_text("新词\n", encoding="utf-8")

            self.assertFalse(word_filter.check("旧词").blocked)
            self.assertTrue(word_filter.check("新词").blocked)

    def test_missing_file_allows_text(self):
        word_filter = SensitiveWordFilter("missing-sensitive-words.txt")

        self.assertFalse(word_filter.check("正常直播互动").blocked)

    def test_directory_loads_multiple_category_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            (root / "adult.txt").write_text("成人词\n", encoding="utf-8")
            (root / "ads.txt").write_text("广告词,推广词\n", encoding="utf-8")
            (root / "custom.txt").write_text("自定义词\n", encoding="utf-8")

            word_filter = SensitiveWordFilter(str(root))

            adult_result = word_filter.check("这里有成人词")
            ads_result = word_filter.check("这里有推广词")
            custom_result = word_filter.check("这里有自定义词")

            self.assertTrue(adult_result.blocked)
            self.assertEqual(adult_result.category, "adult")
            self.assertTrue(ads_result.blocked)
            self.assertEqual(ads_result.category, "ads")
            self.assertTrue(custom_result.blocked)
            self.assertEqual(custom_result.category, "custom")


if __name__ == "__main__":
    unittest.main()
