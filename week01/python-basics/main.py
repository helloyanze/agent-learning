
import json
import sys

from candidate import Candidate


def main() -> None:
    data = json.load(sys.stdin)

    candidate = Candidate.from_dict(data)

    candidate.print_profile()


if __name__ == "__main__":
    main()