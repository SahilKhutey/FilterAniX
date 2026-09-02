import json
from src.core.hardware import system_info


def main():
    info = system_info()
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
