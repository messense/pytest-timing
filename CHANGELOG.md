# Changelog

## 0.1.0

- Output options are boolean flags (`--timing-json`) with separate `--timing-json-file PATH`
  options, so an output flag can never swallow a test path.
- Runs record their termination (`finished`, `interrupted`, `aborted`, ...) explicitly.
- Initial release: controller-side timing capture with and without pytest-xdist,
  ASCII Gantt chart in the terminal summary, self-contained HTML report, JSON output,
  Chrome trace output, and a `pytest-timing` CLI with `render` and `merge` commands.
