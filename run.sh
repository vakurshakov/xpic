#!/bin/bash

usage() { echo "Usage: $0 [-v] <config.json>" 1>&2; exit 1; }

VALID_ARGS=$(getopt --name "run.sh" --options h,v --longoptions help,verbose -- "$@")
if [[ $? != 0 ]]; then
  usage
fi

eval set -- "$VALID_ARGS"
while [[ "$1" != -- ]]; do
  case "$1" in
    -v|--verbose)
      verbose=1
      ;;
    *|--)
      usage
      ;;
  esac
  shift
done
shift

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_NUM_THREAD=16

if [[ $verbose ]]; then
  export OMP_DISPLAY_ENV=verbose
  export OMP_DISPLAY_AFFINITY=true  # to measure thread migration
fi

if [[ $1 == "" ]]; then
  usage
fi

./bin/simulation.out $1
