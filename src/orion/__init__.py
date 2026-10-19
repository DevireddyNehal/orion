import sys
from pathlib import Path


def main() -> None:
    # Ensure project root is in sys.path
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import main as orion_main

    orion_main.main()
