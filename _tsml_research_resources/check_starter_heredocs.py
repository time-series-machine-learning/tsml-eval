"""Compile every quoted python heredoc inside the given shell scripts."""
import ast, re, sys
bad = 0
for path in sys.argv[1:]:
    text = open(path, encoding='utf-8', newline='').read().replace('\r\n', '\n')
    for m in re.finditer(r"<<'(PYTHON|PY)'\n(.*?)\n\1\n", text, re.S):
        try:
            ast.parse(m.group(2))
            print(f"  ok   {path}: {m.group(2).splitlines()[0][:50]}")
        except SyntaxError as e:
            bad += 1
            print(f"  FAIL {path}: line {e.lineno}: {e.msg}")
sys.exit(1 if bad else 0)
