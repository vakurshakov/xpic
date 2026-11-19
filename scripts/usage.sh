# Sourced by the scripts: `usage` prints the header comment of the calling script ($0), so the
# help text exists only there. The header is the comment block right after the shebang.

usage() {
  awk 'NR == 1 || (!seen && /^$/) { next } /^#/ { seen = 1; sub(/^# ?/, ""); print; next } { exit }' "$0"
}
