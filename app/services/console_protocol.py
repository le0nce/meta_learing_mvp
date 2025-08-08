"""Console Protocol for Dependency Injection"""

from typing import Protocol


class ConsoleProtocol(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for console output dependency injection"""

    def print(self, *args, **kwargs) -> None:
        """Print method for console output"""
