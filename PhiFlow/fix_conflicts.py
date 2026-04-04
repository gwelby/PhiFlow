import sys
import re

def fix_git_conflict(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(r'<<<<<<< HEAD\n(.*?)\n=======\n.*?\n>>>>>>> (?:origin/master|master)\n?', re.DOTALL)
    new_content = pattern.sub(r'\1\n', content)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)

for f in ['d:/Projects/PhiFlow-compiler/PhiFlow/src/phi_core.rs', 'd:/Projects/PhiFlow-compiler/PhiFlow/src/parser/mod.rs', 'd:/Projects/PhiFlow-compiler/PhiFlow/src/phi_ir/emitter.rs']:
    fix_git_conflict(f)

print("Fixed conflicts by accepting HEAD.")
