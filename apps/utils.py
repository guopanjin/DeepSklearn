import re

ANSI_ESCAPE = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)

def remove_ansi_codes(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)
