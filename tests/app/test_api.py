"""TEST API endpoints."""

import pytest
from fastapi import status
from httpx import AsyncClient, Response


@pytest.mark.asyncio
async def test_root(client: AsyncClient):
    """Test the default root of the project"""
    response: Response = await client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"msg": "meta_learning_mvp online"}
