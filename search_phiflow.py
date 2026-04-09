from pathlib import Path
import re
patterns = [
    ("IBM hardware runner", re.compile(r"IBM hardware runner", re.IGNORECASE)),
    ("bijective phase map", re.compile(r"bijective phase map", re.IGNORECASE)),
    ("browser shim", re.compile(r"browser shim", re.IGNORECASE)),
]
root = Path('.')
allowed_ext = {'.md', '.txt', '.rs', '.py', '.html', '.json', '.jsonl'}
for path in root.rglob('*'):
    if not path.is_file():
        continue
    if path.name == 'search_phiflow.py':
        continue
    if path.suffix.lower() not in allowed_ext:
        continue
    try:
        text = path.read_text(errors='ignore')
    except Exception:
        continue
    for name, pattern in patterns:
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                print(f"{name}|{path}|{lineno}|{line.strip()}")
