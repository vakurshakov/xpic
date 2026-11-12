# Sourced by all scripts from the repository root. Every variable can be overridden
# from the environment; `STACK=<name>` sources scripts/stacks/<name>.sh first.
# PETSC_ARCH is not defaulted here: CMake picks linux-mpi-debug/-opt per build type unless it is set.

scripts_dir="$(dirname "${BASH_SOURCE[0]}")"

source "$scripts_dir/versions.sh"

stack="${STACK:-local}"

if [[ ! -f "$scripts_dir/stacks/$stack.sh" ]]; then
  echo "Unknown STACK '$stack': no file $scripts_dir/stacks/$stack.sh" >&2
  return 1 2> /dev/null || exit 1
fi

source "$scripts_dir/stacks/$stack.sh"

mpicxx="$(command -v mpicxx)"
export MPI_DIR="${MPI_DIR:-${mpicxx:+$(dirname "$(dirname "$mpicxx")")}}"

export JSON_DIR="${JSON_DIR:-./external/json}"
export PETSC_DIR="${PETSC_DIR:-./external/petsc}"

[[ -n "$GCC_DIR" ]] && export GCC_DIR

unset scripts_dir stack mpicxx
