#!/usr/bin/env bash

# Installs what `xpic` needs: prerequisite tools (through the system package manager), nlohmann::json and PETSc.
# Usage: ./scripts/install-deps.sh [debug|opt] [--install] [--force]
#   debug|opt  build only this PETSc arch (default: both)
#   --install  accept all prompts to install missing tools
#   --force    re-fetch and rebuild dependencies even if the pinned version is present

for arg in "$@"; do
  case "$arg" in
    --install) auto_install=1 ;;
    --force) force=--force ;;
    debug | opt) arch="$arg" ;;
    *) echo "Unknown argument '$arg'. Usage: $0 [debug|opt] [--install] [--force]" >&2; exit 1 ;;
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

# Prints the install command for a package list, MPI is installed from the distribution, never built.
install_command() {
  local sudo=""
  [[ $manager != brew && $EUID != 0 ]] && sudo="sudo "
  case $manager in
    pacman) echo "${sudo}pacman -S --needed $*" ;;
    apt-get) echo "${sudo}apt-get install -y $*" ;;
    dnf) echo "${sudo}dnf install -y $*" ;;
    brew) echo "brew install $*" ;;
  esac
}

# Package providing a tool for the detected manager.
package_of() {
  case "$manager:$1" in
    apt-get:g++) echo g++ ;;
    dnf:g++) echo gcc-c++ ;;
    pacman:g++ | brew:g++) echo gcc ;;
    apt-get:gfortran) echo gfortran ;;
    dnf:gfortran) echo gcc-gfortran ;;
    pacman:gfortran | brew:gfortran) echo gcc ;;
    apt-get:mpicxx) echo "openmpi-bin libopenmpi-dev" ;;
    dnf:mpicxx) echo openmpi-devel ;;
    pacman:mpicxx) echo openmpi ;;
    brew:mpicxx) echo open-mpi ;;
    apt-get:python3 | dnf:python3) echo python3 ;;
    pacman:python3 | brew:python3) echo python ;;
    *) echo "$1" ;;
  esac
}

for tool in git gcc g++ gfortran mpicxx make cmake python3; do
  command -v "$tool" &> /dev/null && continue

  cmd=$(install_command "$(package_of "$tool")")
  if [[ -z "$cmd" ]]; then
    echo "'$tool' is not installed and no supported package manager was found, please install it." >&2
    exit 1
  fi

  if [[ -z "$auto_install" ]]; then
    echo "'$tool' is not installed. Run: $cmd"
    read -r -p "Run it now? [y/N] " answer
    [[ "$answer" == [yY]* ]] || { echo "'$tool' is required, exiting." >&2; exit 1; }
  else
    echo "'$tool' is not installed, running: $cmd"
  fi

  $cmd || exit 1

  # The install can succeed while the tool stays off PATH: on dnf, openmpi-devel puts mpicxx
  # into /usr/lib64/openmpi/bin, which only the environment module adds to PATH.
  command -v "$tool" &> /dev/null || {
    echo "'$tool' is still not on PATH after the install (for dnf, try 'module load mpi/openmpi-x86_64')." >&2
    exit 1
  }
done

./scripts/install-json.sh $force || exit 1
./scripts/install-petsc.sh $arch $force || exit 1
