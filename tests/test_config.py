"""The sample's settings differ from the full study's only in dates and file names, so the sample
runs the same pipeline as the full data."""
import os

import yaml

# in the repo: pipeline.yaml (full study) and pipeline.sample.yaml; in the submission zip,
# tools/package.sh makes the sample's settings pipeline.yaml and keeps the full study's as pipeline.full.yaml
IN_ZIP = os.path.exists("config/pipeline.full.yaml")
FULL = "config/pipeline.full.yaml" if IN_ZIP else "config/pipeline.yaml"
SAMPLE = "config/pipeline.yaml" if IN_ZIP else "config/pipeline.sample.yaml"


def load(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    for key in ["window", "deployment", "splits"]:
        config.pop(key)
    for source in config["sources"].values():
        source.pop("file")
    return config


def test_the_sample_changes_only_dates_and_file_names():
    assert load(SAMPLE) == load(FULL)
