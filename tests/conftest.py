"""Dependencies of tests can be overwritten here"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from app.app import app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Method to overwrite the project dependencies in tests"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
