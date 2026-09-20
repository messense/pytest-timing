# Changelog

## 0.4.0

- Keep slowest-test labels visible at the end of the timeline and wrap long labels
  below their bars when neither side has room.
- Add `pytest-timing compare` with per-test duration, phase, CPU and memory changes,
  optional CI regression budgets, machine-readable JSON and explicit unavailable checks.
- Add shared fixture costs by worker and linked time-range selection in HTML reports.
  Zooming and scrolling resample the visible interval for finer detail.
- Record admission wait intervals on their test attempts, preserve them through
  merging and show them in lanes, a wait graph and range statistics.
- Add `--timing-capture=light` to skip optional CPU and memory measurements while
  retaining durations and resource admission; the default remains full capture.
- Bound HTML chart samples and ticks independently of zoom, retaining peak values
  over each bucket; cache merged-test totals and top-ten hover details.
- Try module/class-contiguous ordering across overlapping fixture families without
  changing worker assignments or increasing modeled CPU work.
- Trace-only output no longer builds an unused JSON report document.
- Runtime fixture admission no longer reserves a dynamic fixture's held CPU slots
  or retained memory twice when it was already included in scheduling history.
- The HTML CPU graph spreads recorded CPU time over the full test span, including
  runtime admission waits, so the graph preserves the recorded total. Test details
  continue to show average CPU use during execution, with those waits excluded.

## 0.3.2

- The HTML report shows CPU and memory usage. The table gains "CPU time", "CPUs"
  (the CPUs a test kept busy on average, beside the slots it declared) and "Memory"
  (what it needed on top of its worker's footprint) columns, the hover details say
  the same with what the measurement covered, the header totals the run and names
  each host's budget, and two graphs show CPUs busy and resident memory over time,
  estimated from the per-test records. All of it appears only where there was a
  reading.

## 0.3.1

- Memory admission no longer charges what session- and package-scoped fixtures keep
  resident. Every worker sets them up once and never lets go, so counting them once
  per lane, and again against every other lane, could only hold tests back or park
  a worker until it was shut down: on a suite with a large session fixture the gate
  left two of eight workers nearly idle and made the run almost three times longer
  while the host had tens of gigabytes free. Their memory is now the worker's
  baseline, which the budget's headroom covers (#7).
- Memory estimates are what a test needs of its own. An attempt that set up shared
  fixtures was charged its whole rise and, again, what the fixtures kept, twice the
  residual; its own need is now the rise beyond what stayed. The first attempt on
  each worker, whose window also covers the worker's warm-up, counts only for a
  test or fixture with no other attempt (#7).
- Waits say which gate held the test. A test the memory gate held back carries the
  seconds in `memory.wait`, beside `cpu.wait` for CPU slots, and the summary line
  reports memory waits even when a CPU budget is set. Time a worker sat parked and
  then left without running what it waited for is reported per gate as well, instead
  of vanishing. The HTML report's table gains a "Held" column and its hover details a
  "held" line, and the terminal's slowest-tests rows say how long each was held (#7).

## 0.3.0

- Estimate a test's cost from the median of its best attempts instead of its
  longest one. The longest attempt grows with the number of attempts, so a history
  merged from several runs drifted upward and a flaky test's worst run was taken as
  its cost. Attempts are ranked clean before contended and passing before failing,
  and fixture set-up costs use the median too.

- Memory-aware admission under pytest-xdist. `--timing-memory SIZE|auto` (ini
  `timing_memory`, env `PYTEST_TIMING_MEMORY`) sets a memory budget per host, and the
  memory each test needed in the run at `--timing-schedule` keeps tests apart whose
  recorded needs would not fit in it together. Nothing is declared: the first run
  records, the next one gates. Memory and CPU budgets are checked together, so a
  test starts only when both fit. A shared fixture keeps what its first test left
  resident reserved while it is alive. The run's JSON gains `memory` with the budget
  and admission summary.
- On macOS, CPU time and memory of live subprocesses are read through `libproc`
  instead of psutil. psutil's child listing scans the whole process table, and the
  CPU clock is read several times per test: with psutil installed, `--timing` on
  3,000 trivial tests took over a minute instead of under a second. psutil is now
  used only where no native reader covers subprocesses, which is Windows.
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
