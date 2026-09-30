# Security notes

* **No secrets.** The project needs no API keys, tokens or credentials and makes no network calls.
  `.gitignore` excludes `.env*`, `*.pem` and `*.key` so none can be committed by accident.
* **No unsafe deserialisation.** No `pickle`, `eval`, `exec`, `subprocess` or `shell=True` anywhere.
  Results are written only as CSV / TXT / JSON / PNG.
* **Input validation.** `Config.validate()` rejects out-of-range settings; the CLI enforces positive
  integers and refuses an `--outdir` that is an existing file. Invalid input exits with code 2.
* **Bounded resources.** Search-space sizes, iterations and population are bounded by the config;
  parallelism is capped by `--n-jobs`.
* **Dependencies** are version-ranged in `requirements.txt`; CI runs on Python 3.10-3.12.

Report vulnerabilities by opening a private issue with the maintainers.
