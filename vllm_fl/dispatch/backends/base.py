# Copyright (c) 2026 BAAI. All rights reserved.

"""
Base backend class for operator implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Backend(ABC):
    """
    Abstract base class for operator backends.

    Each backend provides implementations for a set of operators.
    Backends should implement is_available() to indicate whether
    the backend can be used in the current environment.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available in the current environment.

        Returns:
            True if the backend can be used, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this backend.

        Returns:
            Backend name string.
        """
        pass

    @property
    def vendor(self) -> Optional[str]:
        """
        Get the vendor name for this backend (if applicable).

        Returns:
            Vendor name string, or None for non-vendor backends.
        """
        return None
