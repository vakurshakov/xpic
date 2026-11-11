#!/usr/bin/env bash

# Clones nlohmann::json at the pinned tag into JSON_DIR. Sourced environment
# decides the location. A JSON_DIR outside ./external is reused as is.
# Usage: ./scripts/install-json.sh [--force]
#   --force    re-fetch dependencies even if the pinned version is present

source ./scripts/setenv.sh || exit 1

if [[ "$JSON_DIR" != ./external/* && "$JSON_DIR" != "$PWD/external/"* ]]; then
  [[ -f "$JSON_DIR/include/nlohmann/json.hpp" ]] || {
    echo "JSON_DIR=$JSON_DIR has no include/nlohmann/json.hpp" >&2; exit 1;
  }
  echo "Reusing nlohmann::json from JSON_DIR=$JSON_DIR"
  exit 0
fi

JSON_TAG="$(git -C "$JSON_DIR" describe --tags --exact-match 2> /dev/null)"

if [[ "$1" != "--force" && "$JSON_TAG" == "$JSON_PINNED_TAG" ]]; then
  echo "nlohmann::json $JSON_PINNED_TAG is already installed in $JSON_DIR"
  exit 0
fi

echo "Cloning nlohmann::json $JSON_PINNED_TAG into $JSON_DIR"

JSON_REPO=https://github.com/nlohmann/json.git

rm -rf "$JSON_DIR"
git -c advice.detachedHead=false clone --depth 1 --branch "$JSON_PINNED_TAG" $JSON_REPO "$JSON_DIR" || {
  rm -rf "$JSON_DIR"
  echo "Cloning failed (no network?). Copy nlohmann::json to a directory and set JSON_DIR." >&2
  exit 1
}
