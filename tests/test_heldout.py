from advbug.data.heldout import FixRecord, assign_splits, parse_hunks, reverse_apply, to_examples
from advbug.taxonomy import NO_BUG

FIXED_FILE = '''import os


class Loader:
    def load(self, path, default=None):
        if path is None:
            return default
        with open(path) as f:
            data = f.read()
        return data[:-1]


def helper(x):
    return x + 1
'''

# The fix changed `data[:-2]` to `data[:-1]` inside Loader.load
REPLACE_DIFF = """@@ -8,4 +8,4 @@ class Loader:
         with open(path) as f:
             data = f.read()
-        return data[:-2]
+        return data[:-1]
"""

# The fix added the None guard
INSERT_DIFF = """@@ -4,6 +4,8 @@
 class Loader:
     def load(self, path, default=None):
+        if path is None:
+            return default
         with open(path) as f:
             data = f.read()
         return data[:-1]
"""


def record(diff: str) -> FixRecord:
    return FixRecord("pypibugs", "https://github.com/o/r.git", "abc123", "pkg/loader.py", diff)


def test_reverse_apply_replacement():
    buggy, touched = reverse_apply(FIXED_FILE, parse_hunks(REPLACE_DIFF)[0])
    assert "return data[:-2]" in buggy and "return data[:-1]" not in buggy
    assert touched == [10]
    assert buggy.splitlines()[9].strip() == "return data[:-2]"


def test_reverse_apply_insertion_points_at_following_line():
    buggy, touched = reverse_apply(FIXED_FILE, parse_hunks(INSERT_DIFF)[0])
    assert "if path is None" not in buggy
    assert touched == [6]
    assert buggy.splitlines()[5].strip() == "with open(path) as f:"


def test_to_examples_extracts_method_and_labels():
    bug, clean = to_examples(record(REPLACE_DIFF), FIXED_FILE)
    assert bug.code.startswith("def load(self, path, default=None):")
    assert clean.code.startswith("def load(self, path, default=None):")
    assert bug.code.splitlines()[bug.bug_lines[0] - 1].strip() == "return data[:-2]"
    assert bug.bug_class == "boundary_shift"
    assert clean.bug_class == NO_BUG and clean.bug_lines == ()
    assert bug.pair_id == clean.pair_id


def test_to_examples_none_guard_and_drop_reasons():
    bug, _ = to_examples(record(INSERT_DIFF), FIXED_FILE)
    assert bug.bug_class == "none_guard_omission"
    assert to_examples(record(REPLACE_DIFF), None) == "unavailable"
    assert to_examples(record(REPLACE_DIFF.replace("data[:-1]", "data[:-9]")), FIXED_FILE) == "patch_mismatch"


def test_splits_keep_pairs_together():
    bug, clean = to_examples(record(REPLACE_DIFF), FIXED_FILE)
    a, b = assign_splits([bug, clean])
    assert a.split == b.split and a.split in ("dev", "test")
