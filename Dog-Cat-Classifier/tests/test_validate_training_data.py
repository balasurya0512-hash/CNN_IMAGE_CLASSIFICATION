from pathlib import Path

import pytest

from train_CNN import validate_training_data


def test_validate_training_data_raises_for_tiny_dataset(tmp_path: Path):
    cat_dir = tmp_path / "cat"
    dog_dir = tmp_path / "dog"
    cat_dir.mkdir()
    dog_dir.mkdir()
    (cat_dir / "cat.jpg").write_bytes(b"fake")
    (dog_dir / "dog.jpg").write_bytes(b"fake")

    with pytest.raises(ValueError, match="at least 4 images per class"):
        validate_training_data(tmp_path)
