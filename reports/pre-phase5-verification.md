# Verification before Phase 5

Full stack verification completed on 2026-10-08, using Python 3.11.16:

- Ruff lint and formatting: passed.
- Pytest: 222 passed, none skipped; 49.660 seconds.
- Combined line and branch coverage: 99.38% (1089/1091 lines; 192/198 branches).
- Docker Compose build and startup: passed.
- Real PostgreSQL + pgvector integration: passed, including migrations through 0004,
  concurrent request limiting, vector lookup, and TTL checks.
- Portable Ollama 0.40.0: downloaded from the official GitHub release, extracted,
  and verified against its published SHA-256. Executable version confirmed.
- Both routing models downloaded; real generation passed.
- Exact cache, semantic case-variant cache, bypass, JSON output, authentication,
  budget rejection with Retry-After, sanitized missing-model errors, and metrics passed.
- First inference used qwen2.5:1.5b; short-response fallback used qwen2.5:7b.

Evidence is in stack-verification.json, stack-tests.xml, stack-coverage.xml, and
live-ollama-verification.json. The user ran Docker orchestration from normal
PowerShell because this session cannot access Docker Desktop's named pipe.
Live API checks ran from this session against the running gateway.

To repeat verification from normal PowerShell in this workspace's repository:

```powershell
.\scripts\Start-LocalOllama.ps1
.\scripts\Test-Stack.ps1 -DatabaseOnly
```

The first script starts portable Ollama on localhost and pulls qwen2.5:1.5b and
qwen2.5:7b into the workspace. The second builds Compose and runs the full suite
against an additional disposable PostgreSQL container. It removes only that
temporary container, preserves Compose data, and saves stack-verification.json,
stack-tests.xml and stack-coverage.xml under reports.

Ollama is limited to localhost. Docker-to-host connectivity passed on this machine;
no all-interface binding was needed.

Test-Stack.ps1 without -DatabaseOnly also checks
real inference, exact/semantic caching, bypass, short-response fallback, JSON,
authentication, budget rejection, upstream errors, and metrics. Its short-response
check depends on the model obeying the one-digit prompt; inspect the response if
that assertion fails. The recorded run triggered the expected heuristic.
The semantic smoke check uses changed casing, exercising real MiniLM and pgvector
with a different prompt hash. It does not measure paraphrase quality or false hits.
These functional checks are not Phase 5 benchmarks.

Phase 5 has not started. Functional stack verification passed; work pauses pending
the user's instruction to proceed.
