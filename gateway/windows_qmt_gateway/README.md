# Windows QMT Gateway

Run this package only on the Windows machine that has MiniQMT / QMT and a compatible
`xtquant` installation.

```powershell
uv run qmt-gateway qmt-smoke-test
uv run qmt-gateway serve
```

Do not install unknown `xtquant` packages from PyPI. Set `QMT_XTQUANT_PATH` so the
gateway can add the local QMT-provided package directory to `sys.path`.
