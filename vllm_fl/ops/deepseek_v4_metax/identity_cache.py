# SPDX-License-Identifier: Apache-2.0
"""Identity-keyed weak cache without Tensor.__eq__ or stale-id reuse."""

import weakref


class IdentityCache:
    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        entry = self._data.get(id(key))
        if entry is None or entry[0]() is not key:
            return default
        return entry[1]

    def __setitem__(self, key, value):
        ident = id(key)

        def remove(ref):
            entry = self._data.get(ident)
            if entry is not None and entry[0] is ref:
                self._data.pop(ident, None)

        self._data[ident] = (weakref.ref(key, remove), value)
