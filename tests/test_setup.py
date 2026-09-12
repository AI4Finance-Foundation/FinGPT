import os
import pytest

def test_requirements_exists():
    assert os.path.exists("requirements.txt")

def test_requirements_not_empty():
    with open("requirements.txt", "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        assert len(lines) > 0

def test_requirements_commented_lines():
    with open("requirements.txt", "r") as f:
        for line in f:
            assert "#" in line or line.strip() != ""

def test_setup_runs():
    import subprocess
    result = subprocess.run(["python", "setup.py", "--name"], capture_output=True, text=True)
    assert "FinGPT" in result.stdout
