#!/usr/bin/env bash

# Formats the whole project. Local edits are formatted by the PostToolUse hook
# (.agents/hooks/format.sh), run this only for project-wide style fixes.
#
# Usage: ./scripts/run-format-all.sh [-h|--help] [--install]
#   --install   install clang-format with the system package manager if missing
#   -h, --help  print this help and exit

source "$(dirname "${BASH_SOURCE[0]}")/usage.sh"

for arg in "$@"; do
  case "$arg" in
    -h | --help) usage; exit 0 ;;
    --install) install=1 ;;
    *) echo "Unknown argument '$arg'." >&2; usage >&2; exit 1 ;;
  esac
done

if command -v pacman &> /dev/null; then
  manager=pacman
elif command -v apt-get &> /dev/null; then
  manager=apt-get
elif command -v dnf &> /dev/null; then
  manager=dnf
elif command -v brew &> /dev/null; then
  manager=brew
fi

install_command() {
  local sudo=""
  [[ $manager != brew && $EUID != 0 ]] && sudo="sudo "
  case $manager in
    pacman) echo "${sudo}pacman -S --needed clang" ;;
    apt-get) echo "${sudo}apt-get install -y clang-format" ;;
    dnf) echo "${sudo}dnf install -y clang-tools-extra" ;;
    brew) echo "brew install clang-format" ;;
  esac
}

if ! command -v clang-format &> /dev/null; then
  cmd=$(install_command)
  if [ -z "$cmd" ]; then
    echo "'clang-format' is not installed, please install LLVM/clang-format." >&2
    exit 1
  fi

  if [[ -z "$install" ]]; then
    echo "'clang-format' is not installed. Run: $cmd" >&2
    echo "Or rerun with --install to do it automatically." >&2
    exit 1
  fi

  $cmd || exit 1
fi

find                               \
    src/                           \
    tests/                         \
    -type f                        \
    \( -name '*.h' -o              \
       -name '*.tpp' -o            \
       -name '*.cpp' \)            \
    -exec clang-format -i "{}" \;
