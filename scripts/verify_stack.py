"""Run from a normal terminal: Docker build, disposable PostgreSQL tests, and live Ollama checks.

This is a verification script, not the Phase 5 evaluation harness. It never prints API keys.
"""

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree

import httpx
from dotenv import dotenv_values

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / "reports/stack-verification.json"


def command(args: list[str], env: dict[str, str] | None = None, stream: bool = False) -> str:
    completed = subprocess.run(
        args,
        cwd=PROJECT,
        env=env,
        text=True,
        check=True,
        capture_output=not stream,
        timeout=1800,
    )
    return completed.stdout or ""


def docker_path() -> str:
    installed = shutil.which("docker")
    if installed:
        return installed
    candidate = (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/DockerDesktop/resources/bin/docker.exe"
    )
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError("Docker CLI not found. Install Docker Desktop and reopen your terminal.")


def write_report(report: dict[str, object]) -> None:
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def live_checks(settings: dict[str, str | None]) -> dict[str, object]:
    admin = settings.get("ADMIN_API_KEY")
    if not admin:
        raise RuntimeError("ADMIN_API_KEY is missing from .env")
    base_url = "http://127.0.0.1:8000"
    with httpx.Client(base_url=base_url, timeout=600) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise RuntimeError(
                "Gateway health is degraded. Start Ollama, pull both models, "
                "and check Docker-to-host connectivity using the configured OLLAMA_BASE_URL."
            )
        created = client.post(
            "/admin/keys",
            headers={"Authorization": f"Bearer {admin}"},
            json={
                "name": f"stack-verification-{uuid4().hex[:8]}",
                "requests_per_minute": 100,
                "daily_token_budget": 10_000_000,
            },
        )
        created.raise_for_status()
        client.headers["Authorization"] = f"Bearer {created.json()['api_key']}"
        params = {"model": "auto", "temperature": 0, "seed": 0, "max_tokens": 64}
        prompt = "How does a Python dictionary store data? Answer in one sentence."
        body = params | {"messages": [{"role": "user", "content": prompt}]}

        def completion(payload: dict[str, object]) -> httpx.Response:
            reply = client.post("/v1/chat/completions", json=payload)
            reply.raise_for_status()
            return reply

        print(
            "Running real inference and cache checks (first embedding load may take a while)...",
            flush=True,
        )
        first = completion(body)
        exact = completion(body)
        # MiniLM is uncased: a case variant exercises vector lookup with a different hash.
        semantic = completion(body | {"messages": [{"role": "user", "content": prompt.lower()}]})
        assert first.headers["X-Cache"] == "MISS", "New key's first request must miss"
        assert exact.headers["X-Cache"] == semantic.headers["X-Cache"] == "HIT"
        assert first.json()["choices"] == exact.json()["choices"] == semantic.json()["choices"]
        assert (
            exact.json()["usage"]["total_tokens"] == semantic.json()["usage"]["total_tokens"] == 0
        )
        bypass = completion(body | {"cache": False})
        assert bypass.headers["X-Cache"] == "BYPASS" and bypass.json()["usage"]["total_tokens"] > 0
        fallback = completion(
            params
            | {
                "cache": False,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Reply with exactly one digit and nothing else. What is two plus two?"
                        ),
                    }
                ],
            }
        )
        assert fallback.headers["X-Tier"] == "strong", (
            "Inspect model output if it did not trigger the short-answer heuristic"
        )
        assert fallback.headers.get("X-Escalation-Reason") == "short_response"
        structured = completion(
            params
            | {
                "cache": False,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "user",
                        "content": 'Return a JSON object with one key, answer, set to "Paris".',
                    }
                ],
            }
        )
        assert isinstance(json.loads(structured.json()["choices"][0]["message"]["content"]), dict)
        unauthorized = client.post("/v1/chat/completions", headers={"Authorization": ""}, json=body)
        assert unauthorized.status_code == 401
        budget_key = client.post(
            "/admin/keys",
            headers={"Authorization": f"Bearer {admin}"},
            json={
                "name": "stack-verification-budget",
                "daily_token_budget": 1,
            },
        )
        budget_key.raise_for_status()
        limited = client.post(
            "/v1/chat/completions",
            json=body | {"cache": False},
            headers={"Authorization": f"Bearer {budget_key.json()['api_key']}"},
        )
        assert limited.status_code == 429 and int(limited.headers["Retry-After"]) > 0
        invalid = client.post(
            "/v1/chat/completions",
            json=body | {"model": "gateway-verification-nonexistent-model", "cache": False},
        )
        assert invalid.status_code == 502
        metrics = client.get("/metrics-summary", headers={"Authorization": f"Bearer {admin}"})
        metrics.raise_for_status()
        assert metrics.json()["requests"] >= 9
        return {
            "health": health.json(),
            "real_inference": "passed",
            "exact_cache": "passed",
            "semantic_cache_case_variant": "passed",
            "bypass": "passed",
            "short_response_fallback": "passed",
            "json_output": "passed",
            "authentication": "passed",
            "budget_429_retry_after": "passed",
            "missing_model_sanitized_error": "passed",
            "metrics": "passed",
            "serving_models": {
                "first": first.json()["model"],
                "fallback": fallback.json()["model"],
            },
            "scope": "small functional smoke checks; not latency, quality, or savings benchmarks",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-only",
        action="store_true",
        help="Build/start Compose and test PostgreSQL; skip live Ollama checks",
    )
    args = parser.parse_args()
    docker = docker_path()
    report: dict[str, object] = {"started_at": datetime.now(UTC).isoformat(), "status": "running"}
    write_report(report)
    container = f"gateway-verification-pg-{uuid4().hex[:10]}"
    container_created = False
    password = secrets.token_urlsafe(32)
    env = os.environ.copy()
    env.update({"POSTGRES_PASSWORD": password, "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        command([docker, "info", "--format", "{{.ServerVersion}}"])
        print("Building and starting the gateway + PostgreSQL...", flush=True)
        command([docker, "compose", "up", "--build", "-d"], stream=True)
        report["compose_startup"] = "passed"
        write_report(report)
        command(
            [
                docker,
                "run",
                "--rm",
                "-d",
                "--name",
                container,
                "-e",
                "POSTGRES_PASSWORD",
                "-e",
                "POSTGRES_USER=gateway_test",
                "-e",
                "POSTGRES_DB=gateway_test",
                "-p",
                "127.0.0.1::5432",
                "pgvector/pgvector:pg16",
            ],
            env,
        )
        container_created = True
        for _ in range(60):
            ready = subprocess.run(
                [docker, "exec", container, "pg_isready", "-U", "gateway_test"],
                capture_output=True,
                timeout=10,
            )
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Disposable PostgreSQL did not become ready")
        port = command([docker, "port", container, "5432/tcp"]).strip().rsplit(":", 1)[1]
        env["TEST_DATABASE_URL"] = (
            f"postgresql+asyncpg://gateway_test:{password}@127.0.0.1:{port}/gateway_test"
        )
        scratch = PROJECT.parents[1] / "work/stack-sqlite"
        scratch.mkdir(parents=True, exist_ok=True)
        env["TEST_SQLITE_DIRECTORY"] = str(scratch)
        env["COVERAGE_FILE"] = str(scratch / ".coverage")
        print("Running the complete suite, including live PostgreSQL + pgvector...", flush=True)
        command(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "--tb=short",
                "--cov-report=xml:reports/stack-coverage.xml",
                "--junitxml=reports/stack-tests.xml",
            ],
            env,
            stream=True,
        )
        suite = ElementTree.parse(PROJECT / "reports/stack-tests.xml").getroot().find("testsuite")
        assert (
            suite is not None and int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == 0
        )
        report["tests"] = {
            key: int(suite.attrib[key]) for key in ("tests", "skipped", "failures", "errors")
        }
        report["live_postgres"] = "passed"
        write_report(report)
        if args.database_only:
            report["live_ollama"] = "not run: database-only mode"
        else:
            report["live_ollama"] = live_checks(dict(dotenv_values(PROJECT / ".env")))
        report["status"] = (
            "passed" if not args.database_only else "database checks passed; live Ollama pending"
        )
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = str(exc).replace(password, "[redacted]")
        print(report["failure"], flush=True)
        raise
    finally:
        if container_created:
            # Remove only the unique disposable container created above, never Compose data volumes.
            subprocess.run([docker, "rm", "-f", container], capture_output=True, timeout=30)
        report["finished_at"] = datetime.now(UTC).isoformat()
        write_report(report)
        print(f"Verification report: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
