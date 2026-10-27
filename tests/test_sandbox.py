import pytest

from advbug.sandbox import SandboxPool

SOLUTION = "def add(a, b):\n    return a + b\n"
TESTS = """from solution import add
import pytest

def test_small():
    assert add(1, 2) == 3

def test_negative():
    assert add(-1, -2) == -3

@pytest.mark.parametrize("x", [1, 2])
def test_needs_fixture(x):
    assert False  # skipped: takes arguments

class TestAdd:
    def test_zero(self):
        assert add(0, 0) == 0
"""


@pytest.fixture(scope="module")
def pool():
    with SandboxPool(workers=2, timeout=2.0) as p:
        yield p


def test_original_passes(pool):
    (r,) = pool.run([(SOLUTION, TESTS)])
    assert r.all_passed and r.passed == 3 and r.failed == 0


def test_mutant_fails(pool):
    (r,) = pool.run([(SOLUTION.replace("+", "-"), TESTS)])
    assert not r.all_passed and r.failed == 2  # add(0, 0) still passes


def test_infinite_loop_times_out(pool):
    (r,) = pool.run([("def add(a, b):\n    while True:\n        pass\n", TESTS)])
    assert r.timed_out and not r.all_passed


def test_broken_solution_reports_error(pool):
    (r,) = pool.run([("def add(a, b):\n    return undefined_name\nraise RuntimeError('boom')\n", TESTS)])
    assert r.error and r.error.startswith("solution: RuntimeError")


def test_escape_hatches_are_blocked(pool):
    tests = (
        "import socket, subprocess, os\n"
        "def test_socket():\n    socket.create_connection(('example.com', 80))\n"
        "def test_subprocess():\n    subprocess.run(['true'])\n"
        "def test_remove():\n    os.remove('anything')\n"
    )
    (r,) = pool.run([(SOLUTION, tests)])
    assert r.passed == 0 and r.failed == 3


def test_memory_bomb_is_contained(pool):
    (r,) = pool.run([("def add(a, b):\n    return len(bytearray(8 << 30))\n", TESTS)])
    assert not r.all_passed

