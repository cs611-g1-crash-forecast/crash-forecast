#!/usr/bin/env bash
# Builds the submission zip from a tag: the code exactly as tagged, the sample's settings as
# config/pipeline.yaml (the full study's kept as config/pipeline.full.yaml), and the data sample
# in data/ next to it, where docker-compose mounts it. Run from the crash-forecast folder, with the
# full data in ../data and the stack's image built: tools/package.sh submission-v1
set -euo pipefail

TAG=$1
OUT=build/package
rm -rf build && mkdir -p "$OUT/crash-forecast"

git archive "$TAG" | tar -x -C "$OUT/crash-forecast"
mv "$OUT/crash-forecast/config/pipeline.yaml" "$OUT/crash-forecast/config/pipeline.full.yaml"
mv "$OUT/crash-forecast/config/pipeline.sample.yaml" "$OUT/crash-forecast/config/pipeline.yaml"

docker compose run --rm --no-deps jupyter python tools/make_sample.py "$OUT/data"

(cd "$OUT" && zip -qr "../crash-forecast-$TAG.zip" crash-forecast data)
ls -lh "build/crash-forecast-$TAG.zip"
