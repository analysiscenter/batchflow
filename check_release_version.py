import re
import sys

with open('batchflow/__init__.py', 'r') as f:
    version = "v" + re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)
sys.exit(version != sys.argv[1])
