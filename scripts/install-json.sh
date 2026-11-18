#!/usr/bin/env bash

# Clones nlohmann::json at the pinned tag into JSON_DIR.
# A JSON_DIR outside ./external is reused with only version and header checks.
#
# Usage: ./scripts/install-json.sh [-h|--help] [--force]
#   --force     re-fetch even if the pinned version is present
#   -h, --help  print this help and exit
#
# Environment (also see scripts/setenv.sh):
#   STACK       name of scripts/stacks/<name>.sh sourced first (default: local)
#   JSON_DIR    nlohmann::json location (default: ./external/json)

source "$(dirname "${BASH_SOURCE[0]}")/usage.sh"

for arg in "$@"; do
  case "$arg" in
    -h | --help) usage; exit 0 ;;
    --force) force=1 ;;
    *) echo "Unknown argument '$arg'." >&2; usage >&2; exit 1 ;;
  esac
done

source ./scripts/setenv.sh || exit 1

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
    GIT_ADVICE=0 git init "$JSON_DIR" || exit 1
    GIT_ADVICE=0 git -C "$JSON_DIR" fetch --depth 1 "$JSON_REPO" tag "$JSON_PINNED_TAG" || exit 1
    GIT_ADVICE=0 git -C "$JSON_DIR" checkout "$JSON_PINNED_TAG" || exit 1
  ) || {
    rm -rf "$JSON_DIR"
    echo "Cloning failed (no network?). Copy nlohmann::json to a directory and set JSON_DIR." >&2
    exit 1
  }
else
  echo "nlohmann::json $JSON_PINNED_TAG is already installed in $JSON_DIR"
fi

# Step 3: verify the result.
verify_json "$JSON_DIR" || exit 1

# Step 4: nothing to build, json is header-only.
