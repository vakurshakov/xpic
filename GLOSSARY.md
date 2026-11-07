# xpic

Particle-in-cell simulation of kinetic plasma: charged macro-particles move in electromagnetic fields on a periodic Cartesian grid, and their current closes Maxwell's equations. Everything is driven by one JSON config.

## Language

### Plasma content

**Sort**:
A population of macro-particles sharing charge, mass, macro-particle weight and initial distribution (e.g. electrons, ions).
_Avoid_: Species, particle type, population

**Macro-particle**:
A computational particle that stands for many physical ones; it carries a weight.
_Avoid_: Super-particle, marker

**Kinetic particle**:
A macro-particle described by its full position and momentum, gyrating around the magnetic field.
_Avoid_: Full-orbit particle, point particle

**Drift-kinetic particle**:
A macro-particle described by its guiding center, its parallel momentum and its magnetic moment; the gyration is averaged out.
_Avoid_: Guiding-center particle, DK particle (in prose)

**Guiding center**:
The gyration-averaged position of a charged particle in a magnetic field.

**Larmor vector**:
The offset from the guiding center to the particle position, perpendicular to B.
_Avoid_: Gyroradius vector

**Magnetic moment**:
The adiabatic invariant of a drift-kinetic particle, conserved in the drift-kinetic push.
_Avoid_: mu (in prose)

**Conversion**:
Switching a macro-particle between kinetic and drift-kinetic description at a step boundary.
_Avoid_: Transfer, switching

**Attribute**:
One named per-particle quantity of a sort (position `r`, momentum `p`, ...) with a fixed number of components, stored as a flat array in the particle storage.
_Avoid_: Field (that is the electromagnetic field), property, column

**Shape**:
The weight with which a macro-particle's quantity is gathered from or scattered onto grid nodes, shared by the solver and the diagnostics. Its profile is `shape_function`, `shape_radius` is its extent to one side; the `Shape` class caches the node weights of a particle.
_Avoid_: Form-factor, spline, kernel

### Time and space

**Step**:
One timestep: fields and particles advance from time level n to n+1.
_Avoid_: Iteration, cycle

**Cell**:
One cell of the global grid, the unit of particle ownership, ordering and (planned) collisions.

**World**:
The global periodic grid: domain size, spacing and its decomposition over MPI ranks.

**Rank-local box**:
The block of cells owned by one MPI rank.
_Avoid_: Subdomain, patch

**Snapshot**:
The state of a sort at the start of a step, kept while the step's nonlinear iterations move particles.
_Avoid_: Previous state, backup (that is a restart file)

**Update cells**:
The end-of-step operation that wraps periodic coordinates and hands over particles that changed cell or rank.
_Avoid_: Sorting, migration

### Schemes

**Simulation**:
One concrete field-and-particle model advanced by a step, selected by the `Simulation` config key: `basic`, `ecsim`, `ecsimcorr`, `eccapfim`, `drift_kinetic`.
_Avoid_: Solver, scheme (when naming a concrete one)

**ECSIM**:
Energy-conserving semi-implicit scheme: a linear field solve with a particle-derived mass matrix.

**ECSIM-corr**:
Energy- and charge-conserving extension of ECSIM: an additional field correction restores the discrete Gauss law.

**ECCAPFIM**:
Fully implicit scheme conserving energy and charge: a nonlinear field solve iterated with the particle push.  

**Drift-Kinetic**:
Fully implicit scheme based on a reduced distribution function $f(r, v_\|,\mu)$ -- conserves energy and charge.

**Basic**:
The explicit scheme: Boris push, Esirkepov current, FDTD fields.

### Configuration

**Preset**:
A command executed once at initialisation (set fields, load particles, ...); skipped when the run restarts from a backup.
_Avoid_: Initializer, setup

**Step preset**:
A command executed once every step (e.g. injection, damping).

**Command**:
An action on the simulation outside the physics step, run as a preset or a step preset.

**Diagnostic**:
An output producer run every diagnose period: fields, moments, tables, traces, conservation checks, backups.
_Avoid_: Output, probe

**Loader**:
A command that creates macro-particles of a sort (initial loading or injection).
_Avoid_: Initializer

**Backup**:
A checkpoint a run can restart from.
_Avoid_: Snapshot (that is the step-start state)

**Builder**:
A config-driven factory that turns a JSON entry into a command, diagnostic or simulation.
_Avoid_: Factory

### Profiling

**Stage**:
A broad part of a step (e.g. the field solve, the particle push) timed as a whole; stages do not overlap.
_Avoid_: Section, region

**Event**:
A named sub-step timed inside whichever stage is active when it runs; the same event may appear under several stages. The set of names is open: a few shared ones are conventional (`interpolate`, `push`, `decompose`) and any other name may be registered once and reused.
_Avoid_: Timer, probe

**Phase**:
An event that runs inside a threaded region; its time is the mean over the threads of the team of each thread's busy time, so load imbalance shows up as a deficit against the stage time.
_Avoid_: Thread event

**Profiler**:
The single process-wide place where stages, events and phases are registered, started and stopped, and from which `-log_view` and the `LogView` diagnostic report. Timing is on only when logging is requested.
_Avoid_: Clock, timer

### Conservation

**Charge conservation**:
The discrete continuity equation holds between deposited charge and current (Gauss's law stays satisfied).

**Momentum balance**:
The identity $dP/dt = QE + QVB$ for a sort, exact only for the `basic` Boris centering; for other implementations it holds approximately.

**Energy balance**:
Field plus particle energy changes only by what the scheme exchanges with an external source.
_Avoid_: Energy drift (that is the error, not the property)
