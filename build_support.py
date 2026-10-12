"""Fail package builds when the independently distributed Skill is stale."""
from pathlib import Path
import runpy

from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

# Build backends need not add the source root to sys.path.
check_sync = runpy.run_path(str(Path(__file__).resolve().parent / "scripts/sync_cloud_skill.py"))["check_sync"]


class CheckedBuildPy(build_py):
    def run(self):
        check_sync()
        super().run()


class CheckedSdist(sdist):
    def run(self):
        check_sync()
        super().run()
