import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "API"))

_mock_record = MagicMock()
_mock_record.id = 1

_patcher = patch("app.services.db_services.save_prediction", return_value=_mock_record)
_patcher.start()

from app.db import models  # noqa