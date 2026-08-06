import os
import sys

import pytest

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

os.chdir(BACKEND_DIR)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as c:
        yield c
