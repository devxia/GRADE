import dataclasses
from pathlib import Path
import unittest

import grade
from grade.model_registry import (
    DEFAULT_MODEL_ID, MODEL_REGISTRY, ModelSpec, available_models, get_model_spec,
)


class ModelRegistryTests(unittest.TestCase):
    def test_only_paper_registered(self):
        self.assertEqual(DEFAULT_MODEL_ID, "GRADE-paper")
        self.assertEqual(available_models(), ("GRADE-paper",))
        self.assertEqual(tuple(MODEL_REGISTRY), available_models())

    def test_spec_contract(self):
        spec = get_model_spec(DEFAULT_MODEL_ID)
        self.assertIsInstance(spec, ModelSpec)
        self.assertEqual(spec.model_id, DEFAULT_MODEL_ID)
        self.assertEqual(spec.directory_name, "grade-paper")
        self.assertEqual(spec.local_dir, Path("models") / "grade-paper")
        self.assertIsInstance(spec.description, str)
        self.assertEqual(
            [field.name for field in dataclasses.fields(spec)],
            ["model_id", "directory_name", "description"],
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.directory_name = "other"

    def test_invalid_models_raise_value_error(self):
        for value in (None, True, 1, [], {}, "", "grade-paper", "GRADE-base", "GRADE-medium"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                get_model_spec(value)

    def test_package_version(self):
        self.assertEqual(grade.__version__, "1.2.0")


if __name__ == "__main__":
    unittest.main()
