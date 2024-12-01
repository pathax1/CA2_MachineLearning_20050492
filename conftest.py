import pytest
from datetime import datetime
import os

@pytest.fixture(scope="session")
def config():
    # Configuration details for the base URL
    return {
        "base_url": "https://finance.yahoo.com/"
    }
