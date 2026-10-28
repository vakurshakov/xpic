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

### Conservation

**Charge conservation**:
The discrete continuity equation holds between deposited charge and current (Gauss's law stays satisfied).

**Energy balance**:
Field plus particle energy changes only by what the scheme exchanges with an external source.
_Avoid_: Energy drift (that is the error, not the property)
