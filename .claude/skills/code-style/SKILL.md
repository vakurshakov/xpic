---
name: code-style
description: Code style conventions for xpic - comments, includes, declaration/definition order, formatting. Use when writing or reviewing C++ code.
---

# Code style

Use when writing, reviewing, or cleaning up C++ code. Formatting is enforced by `.clang-format`; the rest is convention.

## General

- Descriptive names. Minimal code: no speculative abstractions, no backwards-compatibility shims, no commented-out code.

## Includes

- A header includes only what its declarations use. Implementation-only includes (`Shape`, `SimpleInterpolation`, ...) go in the `.cpp`.
- Include the narrowest header providing the symbol and drop ones already pulled in transitively (e.g. `point.h`, not `particles.h` + `point.h` + `utils.h`). Don't repeat in the `.cpp` what its own header includes.
- A `.cpp` starts with its own header, then a blank line. `clang-format` regroups and sorts the rest (system, PETSc, nlohmann, then `pch` / `interfaces` / other `"src/..."`), so don't hand-order.
- Trimming a header can break transitive users: do a Debug build afterwards.

## Declaration / definition order

- Order definitions in the `.cpp` exactly as declared in the `.h`; the header is the reference. File-local helpers (anonymous constants, free functions) stay at the top of the `.cpp`.
- Within a class: constructors, accessors, public API, private helpers, then data members.
- When reordering, do pure moves only. Verify with `diff <(sort old) <(sort new)` (empty, ignoring blank lines) and don't reformat the file.


## Comments

- Default to no comment. Keep one only when the WHY is non-obvious: notation of cryptic names (`FRk`, `lenBh_eff`), overload disambiguation, a non-obvious `@warning`, an unreadable type alias.
- Remove a `@brief` that restates the function name or signature (setters, getters, `update_r`, `process_picard`, `decomposition_J`, ...).
- Doxygen `///` (`@brief`, `@warning`, `@note`) only for declarations. One `@brief` per class, never a detached or duplicated one. No `@file` tags.
- Member-variable comments are trailing plain `//`, not `///<`. Group headers are plain `// Group name`, not banner lines (`// -- Group ------`).
- Consecutive trailing comments start in the same column. `clang-format` does this (`AlignTrailingComments`, 2 spaces before `//`).


## Formatting

- Run `clang-format -i` on the files you edit, not the repo-wide `./run-clang-format.sh`, to avoid unrelated churn.
- 80 columns, 2-space indent, `PointerAlignment: Left`, function-definition brace on its own line, class/struct brace on the same line.
- Header guards use `SRC_<PATH>_H` (no `#pragma once`). PETSc error handling follows `PetscFunctionBeginUser` / `PetscCall(...)` / `PetscFunctionReturn(PETSC_SUCCESS)`.
- The build must stay clean under `-Wall -Wextra -Wpedantic -Werror`.

### JSON configuration formatting

In `Configuration::overwrite({...})` calls, trailing commas fine-tune layout after `clang-format`:

- Top-level keys (`"Geometry"`, `"Particles"`, `"Presets"`, `"Diagnostics"`) have trailing commas.
- Items in arrays (particle specs, preset commands, diagnostic entries) have trailing commas.
- Nested objects like `"setter"`, `"coordinate"`, `"momentum"` use **inline format** when the key-value pair fits: `{"setter", {...}}`, not broken across lines.
- When a preset or diagnostic item cannot fit on one line within `ColumnLimit`, break it into multi-line format with trailing commas on sub-keys, but keep nested object assignments inline.
- Last item in nested objects typically has no trailing comma, unless it improves vertical alignment.
- **Check existing test files** in the same directory for formatting patterns before applying changes.

This keeps multi-line JSON compact and prevents `clang-format` from collapsing structure.
