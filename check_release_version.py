import re
import sys

with open('batchflow/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]v([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)
sys.exit(version != sys.argv[1])
