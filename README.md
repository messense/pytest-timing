# pytest-timing

Record when every test ran, on which [pytest-xdist](https://github.com/pytest-dev/pytest-xdist)
worker, and for how long, then render the run as a timing report in the spirit of
`cargo build --timings`: an ASCII Gantt chart in the terminal and a self-contained
HTML report with worker lanes, a concurrency graph and a sortable table.

```
pytest -n 4 --timing --timing-html
```

```
================================ timing report =================================
pytest-timing: 129 tests (1 error, 1 failed), 4 workers, wall 1.37s, busy 95.4%
worker |-----------|-----------|------------|-----------|-----------|--------  busy%
gw0    ░░░░░░   ▄███████████████▄███████████████████████████████████████████    93.6%
gw1    ░░░░░░░  ▄██████████████████████████████████████████████████X            94.1%
gw2    ░░░░░░░░▒▄███████████████████████████████████████████████████▄           96.2%
gw3    ░░░░░░░░░▄████████████████████████████████▄███████████XXXXX             98.2%
       0s          0.20s       0.40s        0.60s       0.80s       1.00s 1.37s
       legend: ░ boot  ▒ collect  █ tests  ▄ <50% busy  X failure

slowest 3 tests (setup ░ / call █ / teardown ▒):
gw0                                                    ████████████████████▒    0.36s
       test_big.py::test_slow[5]
gw2                                               ████████████████▒            0.30s
       test_big.py::test_slow[4]
gw1               ░░░░░░░░░░░░░▒                                               0.21s
       test_big.py::test_many[8]
HTML report written to pytest-timing.html
```

Works with and without `-n`. Without xdist the run is a single `main` lane.

Each lane shows worker boot, collection, tests (`▄` marks a column that is under half busy,
so idle gaps stand out) and failures. The slowest tests are drawn on the same axis with
their setup / call / teardown split; note how the module-scoped fixture above lands in the
setup phase of the first test on each worker.

The HTML report has the same data with a zoomable lane Gantt, a concurrency graph, filters,
hover details, and a sortable table:

![HTML report](docs/report.png)

## Install

```
pip install pytest-timing            # plugin only
pip install "pytest-timing[xdist]"   # with pytest-xdist
```

Python 3.10+, pytest 7.3+ (the version that added wall-clock `start`/`stop` to test reports).

## Usage

| Option | Effect |
|---|---|
| `--timing` | Record timings and print the ASCII chart in the terminal summary. |
| `--timing-html` | Write a self-contained HTML report to `pytest-timing.html`. |
| `--timing-json` | Write the recorded run to `pytest-timing.json`. |
| `--timing-trace` | Write a Chrome trace file for [Perfetto](https://ui.perfetto.dev) to `pytest-timing.trace.json`. |
| `--timing-html-file PATH`, `--timing-json-file PATH`, `--timing-trace-file PATH` | Same, to an explicit path. |
| `--timing-top=N` | Rows in the slowest-tests section (default 10, `0` hides it). |
| `--timing-min=SECONDS` | Hide tests shorter than this from the slowest-tests section. |
| `--timing-ascii-style=unicode\|ascii` | Chart glyphs. |
| `--timing-width=N` | Override the terminal width for the chart. |

Any output option implies `--timing`. The output flags are plain booleans and the `-file`
options always take a path, so `pytest --timing-json test_x.py` runs exactly `test_x.py`.

The same settings are accepted as ini keys (`timing`, `timing_html`, `timing_json`,
`timing_trace` as paths or `true`, plus `timing_top`, `timing_min`, `timing_ascii_style`)
and environment variables (`PYTEST_TIMING=1`, `PYTEST_TIMING_HTML=path`, ...), so CI can
enable it without touching the command line.

Every JSON run records how the session ended (`finished`, `collect_only`, `interrupted`,
`aborted`, `internal_error`) with the reason pytest gave, and `complete` is derived from that.

### View the trace

`--timing-trace` writes a Chrome Trace Event file. Open it in
[Perfetto UI](https://ui.perfetto.dev) with "Open trace file", or in `chrome://tracing`.
Each worker is a track, every test is a bar, and the setup / call / teardown phases nest
underneath it. Zoom with W/A/S/D and select a range to aggregate durations.

### Re-render or merge saved runs

```
pytest-timing render pytest-timing.json --html report.html --ascii
pytest-timing merge shard1.json shard2.json -o all.json
```

`merge` places several runs (for example CI shards) on one shared time axis using their
absolute start times.

## How it works

Since pytest 7.3 every `TestReport` carries wall-clock `start` and `stop` timestamps. xdist
serialises those to the controller unchanged and attaches the worker to the report, so the
plugin only needs controller-side hooks: `pytest_runtest_logreport` for the setup / call /
teardown phases, plus xdist's node-ready, collection-finished and node-down hooks for the
worker lifecycle. Nothing runs inside the workers and nothing touches the execnet channel.

Each lane in the report shows boot (worker start-up until it is ready), collection, tests,
idle gaps and the point the worker shut down. Session-scoped fixture setup is attributed to
the first test's setup phase and its teardown to the last test's teardown phase, exactly as
pytest reports it.

## Overhead

Nothing runs inside the workers: the plugin only listens to the reports xdist already
sends to the controller, and each of the three phase reports per test costs a few
microseconds of bookkeeping. Rendering happens once, at the end of the session.

Measured on 5,000 trivial tests (a worst case, since the per-test cost is fixed while the
tests themselves take almost nothing), best of five runs:

| Configuration | Wall time | Overhead |
|---|---|---|
| single process, plugin disabled | 1.29 s | |
| single process, `--timing` | 1.38 s | +0.09 s |
| single process, `--timing` plus JSON, HTML and trace files | 1.45 s | +0.16 s |
| `-n 4`, plugin disabled | 1.18 s | |
| `-n 4`, `--timing` | 1.25 s | +0.07 s |
| `-n 4`, `--timing` plus all three files | 1.32 s | +0.14 s |

That is under 20 microseconds per test for recording, plus a fixed serialisation cost per
output file of roughly 10 ms per thousand tests. Output size is about 250 bytes per test
for the JSON and HTML files and 600 bytes for the trace. Memory held during the run is on
the same order as the JSON. The plugin registers nothing at all unless one of its options
is enabled.

## License

MIT
