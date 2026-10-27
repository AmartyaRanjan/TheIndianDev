import pytest
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    except OSError:
        pytest.skip("Qwen tokenizer not downloaded")
