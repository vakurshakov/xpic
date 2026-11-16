# Sourced by the cluster install scripts after setenv.sh: checks the inputs, sets JOBS and holds
# the helpers. The steps themselves stay in each script.
# The scripts never download: sources come from TARBALL_DIR.

[[ -n "$STACK" ]] || { echo "STACK is required (e.g. STACK=stark-gcc-openblas)." >&2; exit 1; }
[[ -n "$STACK_ROOT" ]] || { echo "STACK_ROOT is required (e.g. STACK_ROOT=/path/to/stack/root)." >&2; exit 1; }
[[ -d "$TARBALL_DIR" ]] || { echo "TARBALL_DIR is required and must be a directory (got '$TARBALL_DIR')." >&2; exit 1; }

# Scheduler slots, `nproc` outside of it.
JOBS="${JOBS:-${NSLOTS:-$(nproc)}}"

# Exits unless the prefix is free or --force was given.
# Usage: guard_prefix <prefix> <force>
guard_prefix() {
  if [[ -e "$1" && -z "$2" ]]; then
    echo "$1 already exists, pass --force to rebuild it." >&2
    exit 1
  fi
}

# Extracts the tarball from TARBALL_DIR into the component directory, unless the source is there.
# With force the source tree is removed first, so the build starts clean.
# Usage: extract_source <tarball> <component_dir> <source_dir_name> <force>
extract_source() {
  local tarball="$TARBALL_DIR/$1"
  [[ -f "$tarball" ]] || { echo "Missing tarball $tarball." >&2; exit 1; }
  [[ -n "$4" ]] && rm -rf "${2:?}/$3"
  if [[ ! -d "$2/$3" ]]; then
    mkdir -p "$2" && tar -xf "$tarball" -C "$2" || exit 1
  fi
}
