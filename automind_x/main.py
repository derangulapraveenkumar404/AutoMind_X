"""AutoMind-X entry point.

python main.py --quick       ~10 second demo
python main.py               full run
python main.py --help        all options
"""

import sys

from automind.cli import main

if __name__ == "__main__":
    sys.exit(main())
