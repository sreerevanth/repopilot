"""
Tests for the web dashboard (dashboard_server.py).

446 lines with no tests, and three defects in them.

The handler fell through to `SimpleHTTPRequestHandler`, which serves the process
working directory — started from a repository root, `GET /.env` returned the
file. I confirmed a real API key came back over HTTP before fixing it.

Every run value was interpolated into `innerHTML` unescaped: the task string
(under agent-fix.yml that is a GitHub issue body anyone can open), the model's
own prose, and the sandbox's captured stdout and stderr.

And the bind was `("", 8080)` — every interface — while the docstring and the
startup banner both said localhost.
"""

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dashboard_server  # noqa: E402


@pytest.fixture
def server(tmp_path, monkeypatch):
    """
    A live server on an ephemeral port.

    `LOGS_DIR` is resolved from the module's own location rather than the
    working directory, so it is patched explicitly — chdir alone does not move
    it. The working directory is still changed, because that is what the
    removed `SimpleHTTPRequestHandler` fallthrough used to serve from.
    """
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(dashboard_server, "LOGS_DIR", str(tmp_path / "logs"))
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-secret\n")
    (tmp_path / "secret.txt").write_text("do not serve me\n")
    monkeypatch.chdir(tmp_path)

    httpd = HTTPServer(("127.0.0.1", 0), dashboard_server.DashboardHTTPHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=5) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


# ── the working directory is not served ───────────────────────────────────


def test_dotenv_is_not_served(server):
    """
    The exact exploit: this returned the file, API key and all, before the
    fallthrough to SimpleHTTPRequestHandler was removed.
    """
    status, body = get(server, "/.env")

    assert status == 404
    assert "sk-ant-secret" not in body


@pytest.mark.parametrize("path", ["/secret.txt", "/dashboard_server.py", "/logs/"])
def test_no_working_directory_file_is_served(server, path):
    status, _ = get(server, path)

    assert status == 404


def test_the_handler_does_not_fall_through():
    """
    `super().do_GET()` is what served the directory. Its absence is the fix;
    a future refactor restoring it would restore the file read.
    """
    source = (ROOT / "dashboard_server.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    assert "super().do_GET()" not in code


# ── the routes that should work ───────────────────────────────────────────


def test_the_index_is_served(server):
    status, body = get(server, "/")

    assert status == 200
    assert "RepoPilot" in body


def test_api_runs_returns_json(server):
    status, body = get(server, "/api/runs")

    assert status == 200
    assert json.loads(body) == []


def test_api_runs_reads_a_summary(server, tmp_path):
    (tmp_path / "logs" / "r1_summary.json").write_text(
        json.dumps({"run_id": "r1", "outcome": "success", "task": "do a thing"})
    )
    _, body = get(server, "/api/runs")

    assert any(run["run_id"] == "r1" for run in json.loads(body))


def test_an_unknown_run_id_returns_empty_rather_than_raising(server):
    status, body = get(server, "/api/runs/does-not-exist")

    assert status == 200
    assert json.loads(body) is not None


@pytest.mark.parametrize("run_id", ["../../etc/passwd", "..%2f..%2fetc%2fpasswd"])
def test_a_traversing_run_id_stays_inside_logs(server, run_id):
    """
    `split("/")[-1]` discards the traversal before the join. This held before
    the change too — pinned so it keeps holding.
    """
    status, body = get(server, f"/api/runs/{run_id}")

    assert status == 200
    assert "root:" not in body


# ── output escaping ───────────────────────────────────────────────────────


def test_the_escape_helper_is_defined(server):
    _, body = get(server, "/")

    assert "function esc(value)" in body


@pytest.mark.parametrize(
    "value",
    ["run.run_id", "iter.llm_analysis", "iter.execution_stdout",
     "iter.execution_stderr", "summary.task"],
)
def test_untrusted_values_are_escaped(server, value):
    """
    Each of these carries text from outside: an issue body, the model's prose,
    or captured test output. `innerHTML` parses its input as HTML.
    """
    _, body = get(server, "/")

    assert f"${{{value}}}" not in body, f"{value} is interpolated raw"


def test_the_escape_helper_covers_the_dangerous_characters(server):
    _, body = get(server, "/")
    helper = body[body.index("function esc(value)"):][:600]

    for char in ("&", "<", ">", '"', "'"):
        assert f'"{char}"' in helper or f"/{char}/g" in helper


# ── the bind address ──────────────────────────────────────────────────────


def test_the_default_host_is_loopback():
    assert dashboard_server.DEFAULT_HOST == "127.0.0.1"


def test_no_wildcard_bind_remains():
    """`("", PORT)` is 0.0.0.0 — every interface."""
    source = (ROOT / "dashboard_server.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    assert 'TCPServer(("", ' not in code


def test_host_and_port_are_configurable():
    import contextlib
    import io

    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(io.StringIO()) as out:
        dashboard_server.main(["--help"])

    assert "--host" in out.getvalue()
    assert "--port" in out.getvalue()
