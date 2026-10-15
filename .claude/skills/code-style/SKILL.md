---
name: code-style
description: Code style conventions for xpic - comments, includes, declaration/definition order, formatting. Use when writing or reviewing C++ code.
---

# Code style

Use when writing, reviewing, or cleaning up C++ code. Formatting is enforced by `.clang-format`; the rest is convention.

## General

- Descriptive names. Minimal code: no speculative abstractions, no backwards-compatibility shims, no commented-out code.
- Short physics notation (`n_pa`, `n_pp`, `k_par`, `v2_0`) is fine if it improves the readability and each member has a trailing `//` comment at its declaration.
- Zero and integer-valued constants have no `.0` (`PetscReal q = 0;`, `{0, 0, pz}`, `== 0`, `1e-12`). Keep `.0` where it reads better in physical expressions (`2.0 * M_PI`).
- Use `x++` for integer counters (`active++`, `it++`); iterators keep `++it`.
- Index `Vector3R` and arrays directly by `Axis` (`bases[a]`, `T[a]`), without `static_cast<std::size_t>(a)`.
- Prefer a loop over axes to three copy-pasted lines. Inline single-use helpers.
- A ternary lvalue is acceptable for a symmetric branch: `((cdf < u) ? x_lo : x_hi) = x;`.
- Constructors: no `std::move` in the init list for cheap-to-copy types (`box(box)`); no default arguments for parameters the caller must set (an implicit default is implementation-specific); no redundant member initializers (`Vector3R phase;`, not `phase{0.0, 0.0, 0.0}`).

- Write everything in English: comments, identifiers, messages, plot labels, in C++ and Python alike. The only exception is `doc/`.

## Errors and validation

- Validate with `PetscCheckAbort(cond, PETSC_COMM_WORLD, PETSC_ERR_*, "msg")`, not `throw std::runtime_error(...)` (with or without `std::format`).
- Pick the code by meaning: `ARG_WRONG` (bad value), `ARG_WRONGSTATE` (bad count/state), `ARG_OUTOFRANGE`, `ARG_CORRUPT` (invalid computed values), `NOT_CONVERGED` (sampling failed).
- State the condition positively (`PetscCheckAbort(n > 0 && v2 > 0, ...)`), not `if (!(x > 0.0) || ...) throw`.
- Drop checks that other code already guarantees. No retry loops that can't help: check once and abort.
- Messages are printf-formatted (`%s`, `%g`, `.c_str()` for strings). Assume required fields (e.g. `SortParameters::sort_name`) are set: no fallbacks.
- A `PetscCheckAbort` is a logical block: surround it with blank lines.
- `PetscCheckAbort`, `PetscCheck`, `LOG`, `PetscCall`, ... are in `WhitespaceSensitiveMacros` (`.clang-format`), so `clang-format` does not rewrap them. Format them by hand, better to keep everything on the same line; checks are formatted with the message on the new line; multi-line messages may be aligned manually.

## Includes

- A header includes only what its declarations use. Implementation-only includes (`Shape`, `SimpleInterpolation`, ...) go in the `.cpp`.
- Include the narrowest header providing the symbol and drop ones already pulled in transitively (e.g. `point.h`, not `particles.h` + `point.h` + `utils.h`). Don't repeat in the `.cpp` what its own header includes.
- Before adding system inculdes, check the `src/pch.h` file. If the file is present there, _do not_ include it.
- A `.cpp` starts with its own header, then a blank line. `clang-format` regroups and sorts the rest (system, PETSc, nlohmann, then `pch` / `interfaces` / other `"src/..."`), so don't hand-order.
- Trimming a header can break transitive users: do a Debug build afterwards.

## Declaration / definition order

- Order definitions in the `.cpp` exactly as declared in the `.h`; the header is the reference. File-local helpers (anonymous constants, free functions) stay at the top of the `.cpp`.
- Within a class: constructors, accessors, public API, private helpers, then data members.
- For a major class, order the public API by lifecycle rather than alphabetically or by first-use: (1) constructor(s), (2) `finalize()`, (3) the method that is the class's main logical unit, (4) the remaining utilities, ordered by usage logic (setup/load helpers, then per-step helpers, then plain accessors).
- `initialize_implementation()` / `finalize()` pairs that come from overriding a base class's public lifecycle hooks (as in `interfaces::Simulation`) keep their natural init-before-finalize order even when a protected section of other overrides sits declared between them in the header; don't force a literal top-to-bottom match across an access-specifier boundary.
- When reordering, do pure moves only. Verify with `diff <(sort old) <(sort new)` (empty, ignoring blank lines) and don't reformat the file.


## Comments

- Default to no comment. Keep one only when the WHY is non-obvious: notation of cryptic names (`FRk`, `lenBh_eff`), overload disambiguation, a non-obvious `@warning`, an unreadable type alias.
- Remove a `@brief` that restates the function name or signature (setters, getters, `update_r`, `process_picard`, `decomposition_J`, ...).
- An override inherits the base-class doc: document the contract once in `src/interfaces/`, not in each `impls/` header. Explain a non-obvious internal helper in the `.cpp` above its definition, not in the header.
- Doxygen `///` (`@brief`, `@warning`, `@note`) only for declarations. One `@brief` per class, never a detached or duplicated one. No `@file` tags.
- Member-variable comments are trailing plain `//`, not `///<`. Group headers are plain `// Group name`, not banner lines (`// -- Group ------`).
- Consecutive trailing comments start in the same column. `clang-format` does this (`AlignTrailingComments`, 2 spaces before `//`).


## Formatting

- Run `clang-format -i` on the files you edit, not the repo-wide `./run-clang-format.sh`, to avoid unrelated churn.
- 80 columns, 2-space indent, `PointerAlignment: Left`, function-definition brace on its own line, class/struct brace on the same line.
- Header guards use `SRC_<PATH>_H` (no `#pragma once`). PETSc error handling follows `PetscFunctionBeginUser` / `PetscCall(...)` / `PetscFunctionReturn(PETSC_SUCCESS)`.
- Blank lines separate logical blocks (declarations, checks, update steps).
- No `// clang-format off/on` wrappers: restate the code so it is format-stable.
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
