import os
import ast
import re

import pytest

# Extract the is_openai_model function source from utils.py without importing the module
utils_path = os.path.join(os.path.dirname(__file__), "..", "src", "vss_ctx_rag", "utils", "utils.py")
with open(utils_path) as f:
    source = f.read()

module = ast.parse(source)
func_source = None
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name == "is_openai_model":
        func_source = ast.get_source_segment(source, node)
        break

namespace = {"re": re}
exec(func_source, namespace)

is_openai_model = namespace["is_openai_model"]

@pytest.mark.parametrize("model", [
    "gpt-4o",
    "gpt-4.5-preview",
    "gpt-image-1",
    "dall-e-3",
    "davinci-002",
    "babbage-002",
    "codex-mini-latest",
    "chatgpt-4o-latest",
    "tts-1",
    "whisper-1",
    "computer-use-preview",
    "text-embedding-3-small",
    "omni-moderation-latest",
    "o1-preview",
    "o1-mini",
    "o3-mini",
    "o1-pro",
    "o4-mini",
    "01-mini",  # zero prefix variant
])
def test_is_openai_model(model):
    assert is_openai_model(model)

