import os
import unicodedata
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class FilterResult:
    blocked: bool
    word: str = ""
    category: str = ""


class SensitiveWordFilter:
    def __init__(self, words_path: str):
        self.words_path = words_path
        self._lock = Lock()
        self._words = []
        self._signature = None

    def check(self, text: str) -> FilterResult:
        self._reload_if_needed()
        normalized_text = self._normalize(text)
        if not normalized_text:
            return FilterResult(False)

        for word, category in self._words:
            if word and word in normalized_text:
                return FilterResult(True, word, category)
        return FilterResult(False)

    def _reload_if_needed(self):
        signature = self._file_signature()
        if signature == self._signature:
            return

        with self._lock:
            signature = self._file_signature()
            if signature == self._signature:
                return
            self._words = self._load_words()
            self._signature = signature

    def _file_signature(self):
        if os.path.isdir(self.words_path):
            signatures = []
            for filepath in self._word_files():
                try:
                    stat = os.stat(filepath)
                except OSError:
                    continue
                signatures.append((filepath, stat.st_mtime_ns, stat.st_size))
            return tuple(signatures)
        else:
            try:
                stat = os.stat(self.words_path)
                return stat.st_mtime_ns, stat.st_size
            except OSError:
                return None

    def _load_words(self):
        words = {}
        for filepath in self._word_files():
            category = os.path.splitext(os.path.basename(filepath))[0]
            for word in self._load_file_words(filepath):
                normalized = self._normalize(word)
                if normalized and normalized not in words:
                    words[normalized] = category
        return sorted(words.items(), key=lambda item: len(item[0]), reverse=True)

    def _word_files(self):
        if os.path.isdir(self.words_path):
            return [
                os.path.join(self.words_path, filename)
                for filename in sorted(os.listdir(self.words_path))
                if filename.lower().endswith(".txt")
            ]
        return [self.words_path]

    @staticmethod
    def _load_file_words(filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
        except OSError:
            return []

        words = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.extend(word.strip() for word in line.split(","))
        return [word for word in words if word]

    @staticmethod
    def _normalize(text: str):
        text = unicodedata.normalize("NFKC", str(text or ""))
        return "".join(text.casefold().split())


_FILTER_CACHE = {}
_FILTER_CACHE_LOCK = Lock()


def get_sensitive_word_filter(words_path: str):
    with _FILTER_CACHE_LOCK:
        if words_path not in _FILTER_CACHE:
            _FILTER_CACHE[words_path] = SensitiveWordFilter(words_path)
        return _FILTER_CACHE[words_path]
