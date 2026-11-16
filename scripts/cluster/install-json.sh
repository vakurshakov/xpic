#!/usr/bin/env bash

# Unpacks the nlohmann::json tarball from TARBALL_DIR into STACK_ROOT (the cluster has no
# internet, so scripts/install-json.sh cannot clone it) and validates the result. Header-only:
# the unpacked source is the install, so SRC must match JSON_DIR of the stack.
#
# Usage: ./scripts/cluster/install-json.sh [-h|--help] [--force]
#   --force     replace an existing JSON_DIR with a fresh extraction
#   -h, --help  print this help and exit
#
# Environment:
#   STACK        name of scripts/stacks/<name>.sh (required)
#   STACK_ROOT   root of the stack installs (required)
#   TARBALL_DIR  directory with the source tarballs (required)

#$ -S /bin/bash
#$ -N install-json
#$ -V
#$ -cwd
#$ -j y

source ./scripts/usage.sh

for arg in "$@"; do
  case "$arg" in
    -h | --help) usage; exit 0 ;;
    --force) force=1 ;;
    *) echo "Unknown argument '$arg'." >&2; usage >&2; exit 1 ;;
  esac
done

source ./scripts/setenv.sh || exit 1
source ./scripts/cluster/common.sh

# Every install script declares the same variables and goes through the same steps (source,
# build tree, configure, build, install, check); only the commands inside the steps differ.
#   NAME     component directory in STACK_ROOT, holds the source (and the build tree)
#   TARBALL  file in TARBALL_DIR
#   SOURCE   directory the tarball unpacks to, inside the component directory
#   PREFIX   install prefix from the stack file
NAME="json"
TARBALL="json-${JSON_PINNED_TAG#v}.tar.gz"
SOURCE="json-${JSON_PINNED_TAG#v}"
PREFIX="$JSON_DIR"

COMPONENT_DIR="$STACK_ROOT"
SOURCE_DIR="$COMPONENT_DIR/$SOURCE"

# Step 1: guard the prefix and extract the source.
guard_prefix "$PREFIX" "$force"
extract_source "$TARBALL" "$COMPONENT_DIR" "$SOURCE" "$force"

# Step 2: verify the unpacked version (header-only: there is nothing to build).
verify_json "$PREFIX" || exit 1
