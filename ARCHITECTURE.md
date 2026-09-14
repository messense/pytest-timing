# Architecture

How pytest-timing captures timings, times shared fixtures inside xdist workers, and
schedules a run from the previous one. For usage, see the [README](README.md).

## Modules

| Module | Role |
|---|---|
| `plugin.py` | pytest hooks on the controller: options, settings, capture, terminal summary, outputs, and the `pytest_xdist_make_scheduler` hook. |
| `collector.py` | Folds phase reports and worker events into a `Run`. Knows nothing about pytest objects. |
| `model.py` | The recorded run: `Run`, `Worker`, `TestSpan`, `Phase`, and the JSON representation. |
| `outputs.py` | The ASCII, HTML and trace renderers, in one table shared by the plugin and the CLI. |
| `cli.py` | `pytest-timing render` and `pytest-timing merge`. |
| `fixtures.py` | Times shared fixture set-up where tests run and lists dependencies on reports. |
| `schedule.py` | Duration estimates, the fixture-aware cost model and the planner. Free of xdist. |
| `demand.py` | Declared CPU demand: the `timing_cpu` marker, the fixture decorator, the declarations a worker sends, and the meter that records each test's CPU work and resident memory. |
| `admission.py` | One host's CPU budget, reservations, fair waiting line and pressure feedback. Free of xdist and platform reads. |
| `telemetry.py` | Platform reads: affinity and cgroup quota, CPU time and resident memory of a process tree, pressure and throttling. |
| `xdist_scheduler.py` | The xdist scheduler that drives workers with the planner and the admission gate, in `load` and `worksteal` mode. |
| `xdist_compat.py` | Everything that reads pytest-xdist internals (versions, worker detection, why the controller stopped, the custom worker event and controller command). |

## Capturing timings

Since pytest 7.3 every `TestReport` carries wall-clock `start` and `stop` timestamps.
xdist serialises reports to the controller unchanged and attaches the worker to each,
so phase aggregation uses controller-side hooks: `pytest_runtest_logreport` for the
setup / call / teardown phases, plus xdist's node-ready, collection-finished and
node-down hooks for the worker lifecycle. CPU admission adds custom events and commands
through the xdist adapter; timing reports use pytest's normal report transport.

Each lane in the report shows boot (worker start-up until it is ready), collection,
tests, idle gaps and the point the worker shut down. Session-scoped fixture set-up is
attributed to the first test that needs that instance on each worker and its teardown
to the test that finalizes it, exactly as pytest reports it.

Timing collection is enabled by `--timing`, an output, scheduling or CPU-budget
setting. Formatting settings alone do not enable it. The marker is always registered;
when enabled, an xdist worker also registers the fixture timer and CPU meter.

## Timing shared fixtures in the workers

pytest charges a class, module, package or session fixture's set-up to the setup phase
of the first test that needs it on each worker, so a test's recorded duration depends
on where it happened to land. The scheduler needs the two apart: how long the test
itself takes, and which shared fixtures it needs at what price.

`fixtures.py` wraps `pytest_fixture_setup` for the shared scopes and times each
set-up, excluding any set-up nested inside it through `getfixturevalue` (that one is
timed on its own). When a test's setup and call reports are built, the timer attaches
`timing_fixtures` to them: a mapping from fixture key to the seconds this test spent
setting the fixture up, or `None` when the test only uses it. Extra report attributes
survive xdist's serialisation, and the collector merges the two reports into
`TestSpan.fixtures`.

A test's fixtures are gathered from three sources, because none is complete alone:

- the static fixture closure, which names what the signatures ask for;
- the request's resolved fixture definitions, which add what `getfixturevalue` pulled
  in for this test, including inside the test body (hence the call report);
- a per-worker memo of what each shared fixture's set-up requested dynamically. A
  test served from the cache never sees those requests, but it depends on them just
  the same.

A fixture key is `<scope>:<scope node id>:<defined at>::<name>[<params>]`. The
scope node is what pytest shares the instance under: the session (empty), the module,
the class, or the matching package collector. A package fixture without a matching
collector belongs to the session. "Defined
at" tells overrides of the same name apart: the fixture's `baseid` (its conftest
directory or test module), or, for a plugin's fixture, whose `baseid` is empty just
like a root conftest's on pytest 7, the defining module and qualified function name.
This also distinguishes a plugin class registered from that same conftest. The parameter part,
taken from the item's
callspec, tells the instances of a parametrized fixture apart whether the fixture or
the test (`indirect=True`) carries the parameters, since pytest keeps only one
instance alive at a time: `base[1]`. A fixture that depends on a parametrized one,
directly or through other fixtures, is torn down with the instance it was built on
and built again for the next parameter, so those indices are part of its key too:
`derived[base=1]`, or `mixed[0,base=1]` for a fixture with parameters of its own. It
is an index, not a value, so two modules that parametrize the same session fixture
with different values at the same index share a key; the planner is slightly
optimistic there.

## Estimates

`Estimates.from_run` subtracts shared set-up and runtime admission waits from each
attempt's duration to estimate the test's *own* time. Crashed attempts and nonpositive
results are excluded. For each node id it ranks the attempts clean before contended
and passing before failing, and takes the median of the best rank present: a failed
attempt usually stops early, and the longest attempt grows with the number of
attempts, so a history merged from several runs would drift upward. Tests without a
usable estimate get the mean of those estimates, or zero if none exist. Fixture
dependencies are combined across attempts, and each fixture key's set-up cost is the
median of the recorded ones.

A missing or unreadable history file leaves xdist's scheduler in place unless an
explicit CPU budget was requested. With that budget the custom scheduler still runs,
using equal 1 ms test estimates. It supports only `load` and `worksteal`.

## Cost model

A shared fixture makes the cost of a test depend on the worker. Every test has a
*family*, the set of shared fixtures it needs. Fixtures cheaper than a millisecond
are ignored unless they declare CPU demand, in which case they are retained even
without recorded set-up time. A queue's cost is its tests' own durations plus every
set-up the queue forces:

- session fixtures are paid once per worker, as are package fixtures without a
  matching package collector;
- class, module and package fixtures last until the worker leaves their scope,
  including through tests that do not use them. Re-entering the scope pays again;
- a parameter change replaces the fixture and its dependents.

A `Lane` holds one worker's projected finish (`free`), fixture instances at that
point (`fixtures`), and unsent plan. `Costs.charge` computes a single item's duration,
CPU work, peak demand and fixture transition. `Costs.project` folds these into a
`Projection` for a whole sequence. Time and CPU work therefore use one traversal.

The scheduler keeps one `Charge` per dispatched item. It records the holds already
included in the peak, so actual live holds supplement it without double counting.
Charged duration, CPU work and the pre-dispatch fixture checkpoint remain historical
facts even when a runtime request changes the reservation. Stealing refunds the same
record and restores that checkpoint.
Ordinary transfers and steals compare finish times with the same CPU-work floor.

## Planning

`plan` places nonempty fixture families first, then the family without shared
fixtures. Families are ordered by cold CPU work (slot-seconds, including set-ups).
Members are ordered by declared slots, then own duration, both descending. Each
family is tried on 1..n lanes: members are greedily split in that order, putting each
on the chunk with the least own time so far, and the chunks go to lanes chosen by
projected finish and fixture cost.

The placement score is the maximum of the longest lane after placement, the average
lane including work still unplaced, and, with admission enabled, total slot-seconds
divided by the sum of domain admission limits. The average avoids splitting an early
family just because lanes are still empty; the CPU floor counts duplicated heavy set-ups.
A split of a nonempty family must improve the score by more than 1% (`SPLIT_GAIN`).
The family without shared fixtures is spread more evenly on near ties instead.

A bounded balance pass then moves single tests off the tail of the longest lane while
that clearly helps, using the same CPU-work floor. Finally each lane is ordered by
fixture-bound work, declared slots and own duration, then grouped by family in order
of first appearance. This prioritizes fixture reuse and heavy work while leaving
lighter work available at the tail for other workers.

Lane free times are clock readings. The percentage tolerance is applied to time left
from the earliest of them, never to the reading itself, or a large monotonic clock
value would swallow every gain.

`transfer` moves a tail chunk of another lane's unsent plan to a lane that ran dry:
the whole tail run of one family, half of it, or its last test, from every other lane,
keeping the move with the best predicted finish. A move that leaves the prediction
unchanged is still taken, because the receiver is idle and the estimates are only
estimates.

Admission delays themselves are not modelled, so projected finish times remain
approximate while workers wait for slots.

## Driving xdist

`DurationScheduling` subclasses xdist's `LoadScheduling` and keeps its lifecycle
(collection check, crash handling, restarts, shutdown), replacing what goes out and
when. Two constraints of xdist's worker protocol shape it:

- a worker runs a test only once it knows the following one, or that there is none
  (it needs `nextitem` for fixture teardown). A queue's last test waits for another
  test or a shutdown command. CPU admission deliberately uses that wait to keep a
  test from starting until slots are available;
- a worker completes tests in the order it received them, and the controller only
  learns about completions after the fact.

Dispatch charges each test with exactly what the lane's cost function says (test plus
set-ups it forces), and completion refunds exactly that, so the projection of when a
worker will be free stays honest. A worker projected to be done by now is late and is
assumed busy with all its dispatched work from now; one still projected to be busy is
never assumed free before it has run the tests it has not started yet, however long
the running one is taking.

**`--dist load`.** Once every worker has collected, the planner lays out a lane per
worker. Each worker gets its next tests from its own plan: at least two, plus a small
time budget of estimated seconds so runs of trivial tests are batched. The budget is
a tenth of a second at most and shrinks for short suites. Refills happen with
hysteresis (below half the budget, filled to the whole budget) so tiny tests do not
cost a command each. A worker whose plan runs dry takes from other lanes' unsent
plans through `transfer`, which accepts moves that do not worsen the predicted
finish. An idle worker can shut down when no suitable work remains; admission may
keep it waiting with its fixtures alive. Everything not yet sent stays on the
controller, where it can still move.

**`--dist worksteal`.** Without CPU admission, each worker's whole plan is dispatched
in one command, as xdist's own work-stealing scheduler does. With admission, dispatch
is segmented at demand increases as described below. A worker down to its last test
is put on a waiting list, and the controller picks a donor and a tail chunk using the
same cost function as `transfer`. Candidates are the last family in the queue, half
that family, and a chunk sized to the dispatch time budget (or the remaining family
if shorter). Chunks smaller than that batch candidate must shorten the run; larger
ones may leave the prediction unchanged. Ties favor balancing donor and thief.
The chunk is requested through xdist's `steal` command, which is all or nothing from
3.7 on: the worker refuses
unless it still holds every requested test. The running test, the one after it, and
one more as a margin for reports in flight are never requested. One steal is in
flight at a time; a refusal (the donor had moved on) leads to a fresh choice against
the updated bookkeeping, and a receiver that would be left with a single test asks
again or is shut down so it can run it. `tests_finished` stays false while a steal is
in flight.

**Restarts.** When a worker dies, its dispatched-but-unrun tests go back to the pool
and are planned over the surviving lanes together with its unsent plan. If no worker
is left, the tests stay pending without a lane and are planned onto the replacement
when it joins.

## CPU demand

A test's declared weight comes from the closest `timing_cpu` marker (function, class,
module); an invalid marker counts as one slot and fails the test at set-up with the
reason, so a typo cannot silently change the schedule. A fixture's demand is an
attribute the `pytest_timing.cpu` decorator sets on the fixture function, in either
decorator order (pytest 8.4 wraps fixture functions in a definition object; both get
the attribute). The decorator stores metadata directly on the fixture rather than
applying a pytest mark to it.

The controller never sees items, only node ids, and xdist's collection report carries
nothing else. So after collecting, every worker builds a `Declarations` record (slots
per item, the keys of CPU-declared shared fixtures in each item's static closure,
each such fixture's set-up and hold slots, and the host's detected CPU environment)
and sends it as one custom event on its channel, ahead of xdist's own collection
event. xdist raises on unknown events, so `pytest_configure_node` wraps each node's
`process_from_remote` on the instance, before the channel callback is registered, to
take that one event off the stream. The handler runs on execnet's receiver thread and
only stores the record under a lock; the scheduler reads it when it lays out the run,
which happens after the collection event that follows it on the same channel.

`Costs` folds the declarations into the cost model: `slots` per test, set-up and hold
slots per fixture definition (keyed without the parameter index), and declared
fixtures added to their tests' families even before any run has timed them. The
`demand` of a test on a worker is the most it may need at any point there, its own
slots or a set-up it has to pay, plus the holds of the fixtures alive around it. A
function-scoped fixture's declaration is folded into its tests' own slots, since its
set-up runs inside the test's reservation: the test weighs its heaviest stage, each
such set-up next to what the other function-scoped fixtures may hold by then, or
the call with all of them alive. Fixtures no signature names (requested with
`getfixturevalue`) are not in any collected family; the worker still sends every
declared shared fixture definition it knows (`<scope>:<defined at>::<name>`), so a
family recorded by an earlier run and loaded through `--timing-schedule` finds the
demand of such a fixture by definition. Without a recorded run the planner cannot
see them, and admission happens at run time instead (next section).

## Admission

xdist's worker protocol is the gate. A worker runs a test only once it has been told
the following one, or been shut down. So every test sent to a worker is *queued* but
not *granted* until that next message goes out, and the controller sends it only when
the worker's reservation covers everything the worker could then run without another
word: the tests before the tail, or the whole queue after a shutdown. Ordinary test
admission uses existing xdist commands; runtime fixture requests and shutdown
cancellation use the custom messages described below. Pytest's `nextitem` teardown
semantics are preserved for executed tests.

The states of a test:

- *planned* on the controller, in a lane's unsent plan;
- *sent*, the ungranted tail of a worker's queue;
- *granted*, once the message after it went out; the worker's reservation covers it;
- *waiting*, when it is the head of its worker's queue and the reservation cannot be
  raised to cover it: the worker blocks in xdist's queue with fixtures alive, and
  `tests_finished` stays false;
- *parked*, when it is next in a worker's plan but could never fit next to what the
  other workers' fixtures hold: it stays on the controller, the worker runs what
  else it has or waits idle, and the test is tried again at every release and exit;
- *completed* at `runtest_protocol_complete`, after teardown, when the reservation is
  recomputed from what is left;
- *cancelled* when shutdown withdraws an unadmitted test (see below). Steals return
  tests to a plan; worker death reports a started test as crashed and requeues the
  unstarted tail.

A worker's reservation covers the largest recorded peak among its granted tests,
plus live fixture holds not already included in that peak, or only the holds while
it is idle. Raising it uses admission; lowering it always goes through. A rise in
demand behind lighter tests is deferred: the batch stops before the message that
would grant it, the worker runs down to it, and the grant is tried when it is the head. So in
`--dist worksteal` a plan is dispatched in segments that end at the next rise, and a
queue never reserves for its heaviest member while lighter ones run. Stolen tests go
to the front of the taker's plan and out through the same gate.

The waiting line (`Admission`) is ordered by arrival. The head's shortfall is pledged:
other workers may raise their reservations only out of what is left, or when the work
behind the request is projected to end before the head could have had its slots
anyway, computed from the projected release times of the reservations in the way
(EASY backfilling). A request above the limit is clamped to the whole limit and
reported, so an oversized declaration can still run.

Fixtures can keep slots occupied while their worker waits, so they need their own
rule. Before a test is dispatched, its demand on that worker is checked against
what the other workers of the domain hold; if it could never fit while they live it
is passed over (parked) and the worker takes the next test of its plan, or waits idle
when there is none. Ordinary admission respects the limit. After every retry of
the waiting line the scheduler looks for a domain at a standstill, where nothing
runs and nobody in line fits, and decides what gives: a parked worker is shut down
first, so its exit lets go of its own holds and its plan is handed to the workers
that stay at that moment (a worker told to shut down takes nothing more); failing
that, the oldest queued head or runtime request is forced, unless a draining worker
can still release the needed holds. Forced admission can exceed the limit and is
counted in the summary. Budgets are per resource domain: `local` for popen workers,
the host named in the execnet spec otherwise. An explicit integer sets each domain's
budget; `auto` uses the smallest budget detected by its workers. When declarations
enable admission without an explicit budget, the detected budget is raised to at
least the domain's worker count. This floor does not prevent heavy tests or fixture
holds from delaying plain tests.

When xdist shuts every worker down itself (`-x`, `--maxfail`, the restart budget for
crashed workers), it calls each node's `shutdown` directly. The scheduler wraps that
method on every node it is given, so the call passes through it first: what the
worker holds is granted if it fits, and otherwise the tail of its queue, the one test
the worker cannot start without another word, is withdrawn with a `timing_cancel`
command, and the worker-side plugin skips that test's protocol. The worker takes
commands of ours off xdist's command dispatcher on its receiver thread, wrapped per
instance like the event on the controller. A worker that dies while holding a single
unadmitted test had not started it: the test goes back to the pool rather than being
reported as crashed.

A worker blocked inside a runtime fixture request has already started its test;
if it dies there, the test is reported as crashed rather than silently replayed.

A fixture reached only at run time may be absent from the planned reservation.
Before setting up a dynamic fixture that declares multiple slots or a hold, the
worker-side meter sends a `timing_request` event (test, fixture key, request id,
set-up and hold slots) and blocks
until a `timing_grant` command names that request. Request ids are unique within the
worker, so a late grant cannot satisfy a retry of the same fixture. On the controller
the event is re-posted as an in-process event, so the run loop handles it on the main
thread like xdist's own. The scheduler treats it as a rise: while it waits the worker
runs nothing, so its reservation drops to what its fixtures hold. The request
reserves the set-up's slots next to those, through the same gate, waiting line and backfilling as
a queued test; the instance then counts as alive on that worker, its hold added to
everything queued there. Two workers asking at once therefore take turns instead of
deadlocking on the slots of the tests they are blocked in. Each request also carries
the total holds of all live fixtures. The worker tracks successful setup and
finalization at every scope, also sending `timing_holds` events on those transitions.
These actual lifetimes are separate from `Lane.fixtures`, which projects through
prefetched tests and can already describe a different scope or
parameter. Failed setup adds no live hold; a granted setup is not yet a live instance.
Function-scoped holds end with the current test and never become shared fixture
costs. A draining shutdown keeps a blocked test behind the gate, but a live draining
worker's request still participates in deadlock recovery. Workers that can release
holds by finishing are allowed to do so before forcing another request. Worker exit
releases its reservation even when xdist skips scheduler removal after maxfail, so a request
cannot remain blocked by a departed worker. After five minutes a blocked request
cancels its fixture setup and asks to recover the test's original reservation.
Only after that grant does it fail and unwind pytest's incomplete fixture instance;
cleanup and retries therefore retain a reservation and valid fixture bookkeeping.
Collection includes declarations on dynamic-only fixtures, including function
fixtures, when deciding to enable the gate. Declarations alone do not select this
scheduler: a usable history file or an explicit CPU budget is also required.

The scheduler records how long each head waited. Runtime requests and phase reports
carry a private collection index and attempt token, so the plugin sums waits onto
the exact execution's `cpu.wait`, including repeated requests, retries and duplicate
selections. A wait before execution belongs to the first attempt. If the worker dies
before sending any phase report, its unreported crash retains that wait. The worker
also records the part inside the test span
as `runtime_wait`. All open fixture clocks pause during that wait, and test estimates
subtract it along with shared setup. CPU `elapsed` and `work` exclude the wait and
its process-tree CPU work, so the measured rate covers execution. Older records
without `runtime_wait` default to zero; pre-start waits are not subtracted again.
Event handlers are registered before each worker starts, even when another plugin
selects the scheduler: without an admission gate a request is granted immediately.
The cgroup quota behind `auto` is read from the process's own group, walking up to
the mount for the tightest limit, on cgroup v2 (`cpu.max`) and v1 (`cpu.cfs_quota_us`).
Which hierarchy holds the `cpu` controller decides: on a hybrid system a v1 line
naming it in `/proc/self/cgroup` wins over the `0::` v2 line, whose hierarchy has no CPU
controller there. Throttling is read from the same group's `cpu.stat` (`throttled_usec`
on v2, `throttled_time` in nanoseconds on v1).

## Measuring CPU work

`ProcessTreeClock` uses `os.times()` for worker CPU time and, on POSIX, reaped child
CPU time. Where available, it adds live descendants through
`/proc/<pid>/task/*/children` and `/proc/<pid>/stat` on Linux, or psutil otherwise.
Readings account for a waited-for child moving from the live total into the reaped
total. Process discovery is a snapshot, so exits during traversal or descendants
that outlive or detach from their parents can leave gaps. Every record reports its
coverage: `tree`, `reaped` (worker plus waited-for children), or `self` (worker only,
including Windows without psutil).

The meter takes a reading at the start of setup and when the teardown report is made,
and the fixture timer takes readings around shared set-ups, so a record carries the
test's total `work` and the `setup_work` inside set-ups charged to it. Incomplete
tests without a teardown report may have no CPU record. Each record also carries
the host's PSI `some` share and whether the cgroup was throttled during
the test; `Estimates.from_run` treats such attempts as contended and prefers a clean
attempt of the same test when one exists, keeping the contended ones in the file.

## Measuring memory

Memory is recorded so that a later run can keep tests that need a lot of it from
running at the same time; nothing schedules on it yet. `ResidentMemory` reads the
resident set size of the process running the tests and, where they can be listed, of
its live descendants: `/proc/self/statm` and the `/proc` walk shared with CPU
measurement on Linux, `proc_pidinfo` and `proc_listchildpids` through `libproc` on
macOS, `GetProcessMemoryInfo` on Windows for the worker alone. psutil fills in what
the platform readers cannot do, today descendants on Windows; its `children` scans
the whole process table, about ten milliseconds on macOS, so it is never preferred
over a native reader. Descendants are summed, so pages they share count more than
once; the total errs on the large side.

A high-water mark such as `ru_maxrss` never comes down, so it cannot say what one
test needed: only the test that first raised the worker's peak would show anything.
`MemorySampler` therefore polls from a daemon thread while a window is open and
keeps the highest total. The worker's own size is read every 20 ms; it costs about a
microsecond on macOS and ten on Linux. Descendants are listed at most every 100 ms,
and while none are found the interval doubles up to a second, since listing costs an
order of magnitude more (and far more with psutil). Opening or closing a window never
lists descendants by itself, so a fast test costs two readings of its own process, a
few microseconds. The meter opens the window at `pytest_runtest_setup` and closes it
when the teardown report is made, so shared fixture set-ups are charged to the test
that paid for them, as their time is. The window goes on the teardown report as
`timing_memory` and into the JSON as `memory`: `base`, `peak` and `after` in bytes,
and the coverage (`tree` or `self`). A platform with no reading records nothing.
The sampler sleeps between windows and is closed at `pytest_unconfigure`.

A test shorter than the sampling interval is seen only at its edges: its record is
what was resident before and after it, and a buffer allocated and freed inside it
is missed. Faulting in enough memory to matter takes longer than one interval.

`peak - base` is the attempt's rise: what it needed on top of the worker's footprint.
`after - base` is what stayed resident, which for the first test of a session fixture
is roughly the fixture. Both are biased by allocator behaviour: a heap that already
grew for an earlier test can serve a later one without raising RSS, so a rise can
undercount a test whose allocations reuse freed heap, and a freed buffer the allocator
keeps can leave `after` high. Large buffers and subprocess memory, the usual causes of
an out-of-memory kill, are mapped and unmapped directly and measure well.

## Feedback

Admission's pressure feedback moves a domain's limit, never its budget, on evidence
of contention: increasing cgroup quota throttling, or a PSI `some` share at or above
25% while the run's own measured CPU rate is below three quarters of the limit. That
rate sums the latest measured rate of each local worker currently running a test;
idle, departed and remote workers contribute nothing. High PSI alone does not lower
the limit when the run's own measured throughput is high enough.
Three contended samples in a row lower the limit by one slot; eight clean
samples in a row raise it by one, up to the budget; a ten-second cooldown separates
moves; samples are taken at most once a second, on the controller's host only, and
only when the host has the signals. Running tests are never touched: a lower limit
defers the next admissions. Without PSI or cgroup statistics the limit stays at the
budget.

## Evaluation

`tests/test_schedule.py` covers estimates, fixture lifetimes, planning, rebalancing
and gated dispatch with fake workers, plus pytester runs with real workers.
`tests/test_admission.py` exercises reservations, the waiting line, backfilling and
pressure feedback. `tests/test_cpu.py` runs subprocess workloads, cold fixture
set-ups, held slots, crashes and `-x`; `tests/test_runtime_admission.py` covers dynamic
fixtures, runtime waits, cancellation, retries and shutdown recovery.

`benchmarks/cpu_bench.py` generates suites of tiny tests, single-CPU tests, internally
parallel tests with and without deadlines, mixed weights, an expensive session
fixture and background load. It compares configurations using whole-process wall
time (including pytest startup and reporting), slowest heavy test, failures, fixture
set-ups, throttling and pressure. For configurations producing a timing report, its
JSON also includes pytest session time; otherwise that field uses process wall
time. Results depend on the machine and load; the script can also run under a
container CPU quota.
