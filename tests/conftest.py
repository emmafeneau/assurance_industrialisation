import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "API"))

from app.db import models  # noqa — enregistre les tables dans Base.metadata
from app.db.database import create_tables


def pytest_configure(config):  # type: ignore
    create_tables()