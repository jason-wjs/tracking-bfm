# ADR 0001: Standalone mjlab Dependency

## Status

Accepted.

## Context

`tracking-bfm` started as work developed inside a fork of the upstream `mjlab`
repository. The new repository is intended to be a standalone package that can
be installed, tested, and iterated without vendoring upstream `mjlab` source
code.

## Decision

We depend on mjlab as an external package through `pyproject.toml`.

We do not vendor or fork mjlab inside this repository.

We register BFM tasks through the `mjlab.tasks` entry point:

```toml
[project.entry-points."mjlab.tasks"]
tracking_bfm = "tracking_bfm"
```

We may use public `mjlab` modules for task registration, environment configs,
RL runners, viewers, utility functions, and wrappers.

We do not import private mjlab script modules such as `mjlab.scripts._cli`.
Script helpers that are needed by `tracking-bfm` live in
`tracking_bfm.scripts`.

Because `mjlab` is a hard dependency, importing `tracking_bfm` should fail fast
when `mjlab` is not installed or incompatible. The package initializer should
not hide missing hard dependencies.

## Consequences

The repository has a clear seam with upstream `mjlab`: public package
interfaces and entry points. This improves Locality for BFM-specific changes
and avoids carrying a forked copy of unrelated upstream implementation.

Task and CLI compatibility must be maintained in this repository. When a legacy
task identifier is retained, it should be registered as an alias and documented
in `docs/migration.md`.

Upgrading `mjlab` is an explicit dependency update and should be verified with
the package test suite.

## Rejected Alternatives

**Continue direct fork development.** Rejected because BFM-specific behavior
would remain mixed with upstream implementation, making upgrades and review
harder.

**Vendor a copy of `mjlab` under `src/mjlab`.** Rejected because it creates an
implicit fork while pretending to be a package dependency.

**Depend on private `mjlab.scripts.*` helpers.** Rejected because those modules
are not a stable interface for an external package.
