import sys

def fix_git_conflict(filename, keep="head"):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    state = "normal"
    
    for line in lines:
        if line.startswith("<<<<<<< HEAD"):
            state = "in_head"
        elif line.startswith("======="):
            state = "in_theirs"
        elif line.startswith(">>>>>>>"):
            state = "normal"
        else:
            if state == "normal":
                new_lines.append(line)
            elif state == "in_head" and keep == "head":
                new_lines.append(line)
            elif state == "in_theirs" and keep == "theirs":
                new_lines.append(line)

    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

keep = "head"
if sys.argv[1] == "--head":
    keep = "head"
elif sys.argv[1] == "--theirs":
    keep = "theirs"
for f in sys.argv[2:]:
    fix_git_conflict(f, keep)
