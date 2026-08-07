#!/usr/bin/env python3
"""
dashboard.py
A zero-dependency web dashboard to visualize RepoPilot agent run history.
Serves an elegant modern UI on http://localhost:8080.
"""

import http.server
import socketserver
import os
import json
import glob
import sys

PORT = 8080
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

class DashboardHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging to keep console clean
        pass

    def do_GET(self):
        if self.path == "/api/runs":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            runs = []
            if os.path.exists(LOGS_DIR):
                summary_files = glob.glob(os.path.join(LOGS_DIR, "*_summary.json"))
                for file_path in summary_files:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            run_data = json.load(f)
                            runs.append(run_data)
                    except Exception:
                        pass
            
            # Sort runs by timestamp descending
            runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
            self.wfile.write(json.dumps(runs).encode("utf-8"))
            return

        elif self.path.startswith("/api/runs/"):
            run_id = self.path.split("/")[-1]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            run_details = {}
            if os.path.exists(LOGS_DIR):
                jsonl_path = os.path.join(LOGS_DIR, f"{run_id}.jsonl")
                summary_path = os.path.join(LOGS_DIR, f"{run_id}_summary.json")
                
                iterations = []
                if os.path.exists(jsonl_path):
                    try:
                        with open(jsonl_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    iterations.append(json.loads(line))
                    except Exception:
                        pass
                
                summary = {}
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, "r", encoding="utf-8") as f:
                            summary = json.load(f)
                    except Exception:
                        pass
                
                run_details = {
                    "summary": summary,
                    "iterations": iterations
                }
            self.wfile.write(json.dumps(run_details).encode("utf-8"))
            return

        # Main Page
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            
            html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RepoPilot Agent Run Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --success: #10b981;
            --error: #f43f5e;
            --text: #f8fafc;
            --text-secondary: #94a3b8;
            --border: #334155;
            --glass: rgba(30, 41, 59, 0.7);
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
            min-height: 100vh;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2.5rem;
            background: var(--glass);
            padding: 1.5rem 2rem;
            border-radius: 1rem;
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            animation: fadeIn 0.8s ease-out;
        }

        h1 {
            font-weight: 800;
            font-size: 2rem;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 2rem;
            height: calc(100vh - 180px);
            animation: slideUp 0.8s ease-out;
        }

        .sidebar {
            background: var(--card-bg);
            border-radius: 1rem;
            border: 1px solid var(--border);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            overflow-y: auto;
        }

        .main-panel {
            background: var(--card-bg);
            border-radius: 1rem;
            border: 1px solid var(--border);
            padding: 2rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .run-card {
            background: #151f32;
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .run-card:hover {
            transform: translateY(-4px);
            border-color: var(--primary);
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.2);
        }

        .run-card.active {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.1);
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.375rem;
            font-size: 0.75rem;
            font-weight: 600;
            width: fit-content;
        }

        .badge-success { background: rgba(16, 185, 129, 0.2); color: var(--success); }
        .badge-failed { background: rgba(244, 63, 94, 0.2); color: var(--error); }
        .badge-neutral { background: rgba(148, 163, 184, 0.2); color: var(--text-secondary); }

        .iteration-view {
            background: #151f32;
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .iteration-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.75rem;
            margin-bottom: 1rem;
        }

        pre, code {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }

        pre {
            background: #0b0f19;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            border: 1px solid var(--border);
            color: #e2e8f0;
        }

        .flex-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
        }

        .stat-box {
            background: #151f32;
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1rem;
            flex: 1;
            text-align: center;
        }

        .stat-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--primary);
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>RepoPilot Dashboard</h1>
            <p style="color: var(--text-secondary)">Autonomous developer agent runs and diagnostics</p>
        </div>
        <div id="connection-status" class="badge badge-success">Live Updates</div>
    </header>

    <div class="dashboard-grid">
        <div class="sidebar" id="runs-list">
            <p style="color: var(--text-secondary)">No runs found</p>
        </div>
        <div class="main-panel" id="detail-panel">
            <h2 style="color: var(--text-secondary)">Select an agent run to view execution details</h2>
        </div>
    </div>

    <script>
        async function fetchRuns() {
            try {
                const response = await fetch('/api/runs');
                const runs = await response.json();
                const runsList = document.getElementById('runs-list');
                runsList.innerHTML = '';
                
                if (runs.length === 0) {
                    runsList.innerHTML = '<p style="color: var(--text-secondary)">No runs found</p>';
                    return;
                }

                runs.forEach(run => {
                    const card = document.createElement('div');
                    card.className = `run-card`;
                    card.dataset.id = run.run_id;
                    
                    const isSuccess = run.outcome === 'success';
                    const badgeClass = isSuccess ? 'badge-success' : 'badge-failed';
                    
                    card.innerHTML = `
                        <div class="flex-row">
                            <span style="font-weight: 600; font-size: 0.95rem; word-break: break-all;">${run.run_id}</span>
                            <span class="badge ${badgeClass}">${run.outcome.toUpperCase()}</span>
                        </div>
                        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem;">Task: ${run.task ? run.task.substring(0, 60) + '...' : 'N/A'}</p>
                    `;
                    
                    card.addEventListener('click', () => {
                        document.querySelectorAll('.run-card').forEach(c => c.classList.remove('active'));
                        card.classList.add('active');
                        viewRunDetails(run.run_id);
                    });
                    
                    runsList.appendChild(card);
                });
            } catch (err) {
                console.error("Error fetching runs:", err);
            }
        }

        async function viewRunDetails(runId) {
            const panel = document.getElementById('detail-panel');
            panel.innerHTML = '<p>Loading run details...</p>';
            
            try {
                const response = await fetch(`/api/runs/${runId}`);
                const data = await response.json();
                const summary = data.summary;
                const iterations = data.iterations;

                let iterationsHtml = '';
                iterations.forEach((iter, idx) => {
                    let changesHtml = '';
                    if (iter.changes_attempted && iter.changes_attempted.length > 0) {
                        iter.changes_attempted.forEach(chg => {
                            changesHtml += `
                                <div style="margin-top: 1rem; border-left: 3px solid var(--primary); padding-left: 1rem;">
                                    <strong>File:</strong> <code>${chg.path}</code> [${chg.action.toUpperCase()}]
                                    <p style="font-size: 0.85rem; color: var(--text-secondary)">${chg.explanation}</p>
                                </div>
                            `;
                        });
                    }

                    iterationsHtml += `
                        <div class="iteration-view">
                            <div class="iteration-header">
                                <h3>Iteration ${iter.iteration}</h3>
                                <span class="badge badge-neutral">${iter.timestamp}</span>
                            </div>
                            <p><strong>Analysis:</strong> ${iter.llm_analysis}</p>
                            <p style="margin-top: 0.5rem;"><strong>LLM Confidence:</strong> ${(iter.llm_confidence * 100).toFixed(1)}%</p>
                            
                            ${changesHtml}
                            
                            <div style="margin-top: 1rem;">
                                <strong>Execution Command:</strong> <code>${iter.execution_command || 'None'}</code>
                                <p style="margin-top: 0.5rem;"><strong>Test Result:</strong> 
                                    <span class="badge ${iter.execution_success ? 'badge-success' : 'badge-failed'}">
                                        ${iter.execution_success ? 'PASSED' : 'FAILED'}
                                    </span>
                                </p>
                                ${iter.execution_stdout ? `<pre style="margin-top: 0.5rem;">${iter.execution_stdout}</pre>` : ''}
                                ${iter.execution_stderr ? `<pre style="margin-top: 0.5rem; border-color: var(--error)">${iter.execution_stderr}</pre>` : ''}
                            </div>
                        </div>
                    `;
                });

                panel.innerHTML = `
                    <div class="flex-row" style="align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem;">
                        <div>
                            <h2>Run Summary: ${summary.run_id || runId}</h2>
                            <p style="color: var(--text-secondary)">Outcome: <span class="badge ${summary.outcome === 'success' ? 'badge-success' : 'badge-failed'}">${(summary.outcome || 'N/A').toUpperCase()}</span></p>
                        </div>
                    </div>

                    <div class="flex-row">
                        <div class="stat-box">
                            <div style="color: var(--text-secondary); font-size: 0.9rem">Iterations Used</div>
                            <div class="stat-value">${summary.iterations_used || 0}</div>
                        </div>
                        <div class="stat-box">
                            <div style="color: var(--text-secondary); font-size: 0.9rem">Task Success</div>
                            <div class="stat-value" style="color: ${summary.outcome === 'success' ? 'var(--success)' : 'var(--error)'}">${summary.outcome === 'success' ? 'YES' : 'NO'}</div>
                        </div>
                        <div class="stat-box">
                            <div style="color: var(--text-secondary); font-size: 0.9rem">Run ID</div>
                            <div class="stat-value" style="font-size: 1.1rem; padding: 0.5rem 0;">${runId}</div>
                        </div>
                    </div>

                    <div>
                        <h3 style="margin-bottom: 1rem;">Task</h3>
                        <p style="background: #151f32; padding: 1rem; border-radius: 0.5rem; border: 1px solid var(--border);">${summary.task || 'N/A'}</p>
                    </div>

                    <div>
                        <h3 style="margin-bottom: 1rem;">Execution History</h3>
                        ${iterationsHtml}
                    </div>
                `;
            } catch (err) {
                console.error("Error loading run details:", err);
                panel.innerHTML = `<p style="color: var(--error)">Error loading run details: ${err}</p>`;
            }
        }

        // Initial Load
        fetchRuns();
    </script>
</body>
</html>
"""
            self.wfile.write(html_content.encode("utf-8"))
            return

        super().do_GET()

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DashboardHTTPHandler) as httpd:
        print(f"\n==================================================")
        print(f"  RepoPilot Dashboard Server started successfully!")
        print(f"  Url: http://localhost:{PORT}")
        print(f"==================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Dashboard Server...")
            sys.exit(0)

if __name__ == "__main__":
    run_server()
