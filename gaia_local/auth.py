import os
from pathlib import Path

from huggingface_hub import login


def maybe_login_from_token_file() -> None:
    env_token = os.environ.get("HF_TOKEN")
    token_path = Path("token.txt")
    token = env_token
    if token is None and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    if token:
        login(token=token, add_to_git_credential=False)