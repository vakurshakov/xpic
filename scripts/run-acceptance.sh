#!/usr/bin/env bash

# Clean-container acceptance run of the README sequence: install-deps, build, smoke tests, run,
# idempotency and the PETSC_DIR reuse path. Needs docker; the working tree is copied in, so
# uncommitted changes are tested and the host's external/, build/ and results/ are untouched.
# Usage: ./scripts/run-acceptance.sh [ubuntu|arch]   (default: ubuntu)

if [[ -n "$XPIC_ACCEPTANCE_INSIDE" ]]; then
  set -eo pipefail

  step() { printf '\n==== [%s] %s ====\n' "$(date +%T)" "$*"; }
  fail() { echo "ACCEPTANCE FAILED: $*" >&2; exit 1; }

  # Containers ship without package indexes; mpiexec refuses to run as root otherwise.
  export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  if command -v apt-get &> /dev/null; then
    apt-get update
  else
    pacman -Syu --noconfirm
  fi

  # Not installed by install-deps.sh, but the *_plot smoke tests (Python validation of the generated data) need them.
  if command -v apt-get &> /dev/null; then
    apt-get install -y python3-numpy python3-matplotlib python3-scipy
  else
    pacman -S --noconfirm --needed python-numpy python-matplotlib python-scipy
  fi

  mkdir /work && tar -C /src --exclude=./external --exclude=./build --exclude=./results --exclude=./.git -cf - . | tar -C /work -xf -
  cd /work

  step "1. install-deps.sh (PETSc debug+opt, make check for both)"
  ./scripts/install-deps.sh --install

  step "2. install-deps.sh again must be a no-op"
  ./scripts/install-deps.sh --install 2>&1 | tee /tmp/second-install.log
  grep -qE 'Cloning|Building PETSc' /tmp/second-install.log && fail "second install-deps.sh did something"

  step "3. build.sh"
  ./scripts/build.sh

  step "4. run-tests.sh -L smoke"
  ./scripts/run-tests.sh -L smoke --output-on-failure

  step "5. run.sh examples/basic.json"
  ./scripts/run.sh examples/basic.json
  [[ -n "$(find results/basic -type f -size +0 2> /dev/null | head -1)" ]] || fail "results/basic has no non-empty output"

  step "6. PETSC_DIR reuse: symlink outside ./external, nothing is built"
  ln -s /work/external/petsc /opt/petsc-reused
  rm -rf build
  PETSC_DIR=/opt/petsc-reused ./scripts/install-deps.sh --install 2>&1 | tee /tmp/reuse-install.log
  grep -q 'Reusing PETSc' /tmp/reuse-install.log || fail "PETSC_DIR was not reused"
  grep -qE 'Cloning|Building PETSc' /tmp/reuse-install.log && fail "reuse path built something"
  PETSC_DIR=/opt/petsc-reused ./scripts/build.sh Release
  readelf -d build/Release/xpic.out | grep RUNPATH | tee /tmp/runpath.txt
  grep -q '/opt/petsc-reused' /tmp/runpath.txt || fail "xpic.out does not use PETSC_DIR=/opt/petsc-reused"

  step "7. smoke tests against the reused PETSc"
  PETSC_DIR=/opt/petsc-reused ./scripts/run-tests.sh -L smoke --output-on-failure

  step "ACCEPTANCE PASSED"
  exit 0
fi

case "${1:-ubuntu}" in
  ubuntu) image=ubuntu:24.04 ;;
  arch) image=archlinux:latest ;;
  *) echo "Unknown distro '$1'. Usage: $0 [ubuntu|arch]" >&2; exit 1 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

exec docker run --rm -v "$PWD:/src:ro" -e XPIC_ACCEPTANCE_INSIDE=1 "$image" bash /src/scripts/run-acceptance.sh
