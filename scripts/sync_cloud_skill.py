"""Generate the independent Skill client from the single maintained Cloud module."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    b"#!/usr/bin/env python3\n"
    b"# Generated from grade/cloud.py by scripts/sync_cloud_skill.py. Do not edit directly.\n"
)
DESTINATION = Path("skills/grade-cloud/scripts/grade_predict.py")


def expected_script(root=ROOT):
    return HEADER + (root / "grade/cloud.py").read_bytes()


def check_sync(root=ROOT):
    destination = root / DESTINATION
    if not destination.is_file() or destination.read_bytes() != expected_script(root):
        raise RuntimeError("Cloud Skill script is out of date. Run python scripts/sync_cloud_skill.py.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check consistency without writing files.")
    args = parser.parse_args()
    if args.check:
        check_sync()
        print("Cloud Skill script is synchronized.")
    else:
        (ROOT / DESTINATION).write_bytes(expected_script())
        print(f"Updated {DESTINATION}")


if __name__ == "__main__":
    main()
