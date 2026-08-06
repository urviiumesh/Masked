import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTING = os.path.join(ROOT, "testing")
BACKEND = os.path.join(ROOT, "backend")
FRONTEND_TESTS = os.path.join(TESTING, "frontend")


def run(cmd, cwd=None, env=None):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = BACKEND + os.pathsep + env.get("PYTHONPATH", "")

    python = sys.executable
    run(
        [
            python,
            "-m",
            "pytest",
            os.path.join(TESTING, "backend"),
            "-v",
            "--tb=short",
        ],
        cwd=TESTING,
        env=env,
    )

    npm = "npm.cmd" if os.name == "nt" else "npm"
    if not os.path.isdir(os.path.join(FRONTEND_TESTS, "node_modules")):
        run([npm, "install"], cwd=FRONTEND_TESTS)
    run([npm, "test"], cwd=FRONTEND_TESTS)
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
