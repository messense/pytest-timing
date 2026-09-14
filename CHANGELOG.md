# Changelog

## Unreleased

- Record each test's resident memory. A sampler thread in the process running the
  tests polls the resident set size of the worker (and, on Linux, macOS or with
  psutil, its live subprocesses) while a test runs. Test JSON records now include `memory` when
  the platform provides a reading: `base` before the test's set-up, `peak` during
  it and `after` at its teardown, in bytes, with the measurement coverage. Nothing
  schedules on it yet.

## 0.2.0

- Raise the minimum supported pytest-xdist version to 3.7. Running without xdist
  remains supported.
- CPU-aware scheduling under pytest-xdist. `@pytest.mark.timing_cpu(N)` declares that a
  test's workload, subprocesses included, needs `N` CPU slots (class and module markers
  set defaults, the closest wins); `@pytest_timing.cpu(N, hold=M)` declares a fixture's
  set-up demand and what it keeps busy while alive. `--timing-cpus N|auto` (ini
  `timing_cpus`, env `PYTEST_TIMING_CPUS`) sets the budget per host, `auto` from the
  affinity mask and cgroup quota (v2 and v1, the process's own group and its
  ancestors), enforced even below the number of workers. In `load` and `worksteal`
  mode, the controller normally admits a test when its reservation fits; a worker
  whose next test does not fit waits with its fixtures alive. The oldest request
  has priority, with backfilling for short work. A test that cannot fit next to other
  workers' fixture holds stays on the controller until those holds end, and demand above
  the admission limit is clamped to that limit. Deadlock recovery can force a request
  above the limit, counted in the summary. Under `-x`/`--maxfail` a test a worker holds
  without having been admitted is withdrawn rather than run. A declared fixture
  reached only through `getfixturevalue` is admitted at run time: the worker asks
  for its slots before the set-up and waits for them. A fixture built on a
  parametrized one is keyed by that parameter too (`derived[base=1]`), since pytest
  sets it up again for every parameter. A plugin's fixture is keyed by its qualified
  function so it never collides with a root conftest override. Test JSON records now include
  `cpu` when available: elapsed time, CPU time of the worker and its
  descendants, declared demand, measurement coverage, host pressure and throttling, and
  the time it waited for slots. Runtime waits are recorded separately and excluded
  from test and fixture duration estimates and CPU rate; a request timeout cancels
  setup and recovers the test's reservation before cleanup. Admission tracks holds
  through unused tests in the same scope, expires package state when leaving the
  package, and releases exited workers' reservations under `--maxfail`.
  On the controller's Linux host, sustained quota throttling or CPU pressure with
  low throughput lowers the limit for new admissions one slot at a time, with gradual
  recovery. Running reservations are unchanged.
- `--timing-schedule PATH` (ini `timing_schedule`, env `PYTEST_TIMING_SCHEDULE`) plans
  pytest-xdist workers' queues from a previous run's JSON: test duration and declared
  CPU demand balanced across workers, shared fixture set-ups counted where they will
  be paid, families of tests that share expensive fixtures kept together and split
  only when that finishes sooner, and unsent work moved
  between workers as the run drifts from the plan. With `--dist worksteal` the plan is
  dispatched whole unless CPU admission requires segments, and rebalanced through
  xdist's steal command. A missing or unreadable history file falls back to xdist's
  scheduling unless an explicit CPU budget keeps admission active with equal test
  estimates. Other distribution modes always keep xdist's scheduler, with a note in
  the summary.
- Shared (class, module, package, session) fixture set-ups are timed in the process that
  runs the tests and recorded per test in the JSON as `fixtures`, keyed by scope, scope
  node, definition and parameter, with the seconds spent on set-ups charged to that test.
  Fixtures requested with `getfixturevalue` count for every test that uses them, and
  `parametrize(..., indirect=True)` parameters are told apart like a fixture's own.

## 0.1.0

- Output options are boolean flags (`--timing-json`) with separate `--timing-json-file PATH`
  options, so an output flag can never swallow a test path.
- Runs record their termination (`finished`, `interrupted`, `aborted`, ...) explicitly.
- Initial release: controller-side timing capture with and without pytest-xdist,
  ASCII Gantt chart in the terminal summary, self-contained HTML report, JSON output,
  Chrome trace output, and a `pytest-timing` CLI with `render` and `merge` commands.
