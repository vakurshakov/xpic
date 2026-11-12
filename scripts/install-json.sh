#!/usr/bin/env bash

# Clones nlohmann::json at the pinned tag into JSON_DIR. Sourced environment decides the location.
# A JSON_DIR outside ./external is reused with only version and header checks.
# Usage: ./scripts/install-json.sh [--force]
#   --force    re-fetch even if the pinned version is present

source ./scripts/setenv.sh || exit 1

for arg in "$@"; do
  case "$arg" in
    --force) force=1 ;;
    *) echo "Unknown argument '$arg'. Usage: $0 [--force]" >&2; exit 1 ;;
  esac
done

# Step 1: a JSON_DIR outside ./external is reused, only verified.
if [[ "$JSON_DIR" != ./external/* && "$JSON_DIR" != "$PWD/external/"* ]]; then
  verify_json "$JSON_DIR" || exit 1
  echo "Reusing nlohmann::json from JSON_DIR=$JSON_DIR"
  exit 0
fi

# Step 2: fetch the pinned tag unless it is already there (or with --force).
JSON_TAG="$(git -C "$JSON_DIR" describe --tags --exact-match 2> /dev/null)"
JSON_REPO="https://github.com/nlohmann/json.git"

if [[ -n "$force" || "$JSON_TAG" != "$JSON_PINNED_TAG" ]]; then
  echo "Cloning nlohmann::json $JSON_PINNED_TAG into $JSON_DIR"

  rm -rf "$JSON_DIR"
  (
    # Not `git clone --branch`: for an annotated tag it warns "is not a commit!".
    git --no-advice init "$JSON_DIR" || exit 1
    git --no-advice -C "$JSON_DIR" fetch --depth 1 "$JSON_REPO" tag "$JSON_PINNED_TAG" || exit 1
    git --no-advice -C "$JSON_DIR" checkout "$JSON_PINNED_TAG" || exit 1
  ) || {
    rm -rf "$JSON_DIR"
    echo "Cloning failed (no network?). Copy nlohmann::json to a directory and set JSON_DIR." >&2
    exit 1
  }
else
  echo "nlohmann::json $JSON_PINNED_TAG is already installed in $JSON_DIR"
fi

# Step 3: nothing to build, json is header-only.

# Step 4: verify the result.
verify_json "$JSON_DIR" || exit 1
