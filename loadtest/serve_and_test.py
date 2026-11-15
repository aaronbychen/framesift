"""Start the API with a given serving config, run loadtest/run_loadtest.py against it, stop it.

    python loadtest/serve_and_test.py --env SERVING_MODE=sequential -- \
        --label before --concurrency 1 4 8 --scans-per-stream 2 --out artifacts/loadtest/before.json
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", nargs="*", default=[], help="KEY=VALUE pairs for the server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log", default=None)
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="args after -- go to run_loadtest.py")
    args = ap.parse_args()

    env = dict(os.environ)
    env.update(kv.split("=", 1) for kv in args.env)
    log = open(args.log or os.devnull, "w")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                               "--port", str(args.port)], env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{args.port}"
    try:
        deadline = time.time() + 600
        while True:
            if server.poll() is not None:
                raise SystemExit(f"server exited with {server.returncode}, see {args.log}")
            try:
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                raise SystemExit("server did not become healthy in 10 min")
            time.sleep(2)
        rest = [a for a in args.rest if a != "--"]
        code = subprocess.call([sys.executable, str(Path(__file__).with_name("run_loadtest.py")),
                                "--url", url, *rest])
        sys.exit(code)
    finally:
        server.terminate()
        try:
            server.wait(30)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
