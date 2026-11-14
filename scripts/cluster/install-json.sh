#!/usr/bin/env bash

# Unpacks the nlohmann::json tarball from TARBALL_DIR into STACK_ROOT (the cluster has no
# internet, so scripts/install-json.sh cannot clone it) and validates the result.
# The tarball's top directory (json-<version>) is kept and must match JSON_DIR of the stack.
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

guard_prefix "$JSON_DIR" "$force"
extract_source "$JSON_TARBALL" "$STACK_ROOT" "$(basename "$JSON_DIR")" "$force"

verify_json "$JSON_DIR" || exit 1
