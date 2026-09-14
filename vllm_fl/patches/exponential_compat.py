# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility shim for Tensor.exponential_(generator=...).

vLLM's TopKTopP sampler calls ``q[i].exponential_(generator=generator)`` to
honor per-request seeds. On some vendor stacks (observed with the Enflame
torch_gcu 2.11 bundle: flag_gems registers an aten::exponential_ Python
kernel once the device backend activates, and that kernel's signature has
no ``generator`` parameter), both the bound builtin method and
``torch.ops.aten.exponential_`` raise
``TypeError: exponential_() got an unexpected keyword argument 'generator'``,
which kills engine startup during the dummy sampler run.

Replace the method with a Python wrapper that tries the generator-aware
aten call first and falls back to the generator-less call (drawing from the
device's global RNG) with a one-time warning when the vendor kernel does not
accept the argument.
"""

import torch

_installed = False
_warned = False


def _exponential_compat(self, lambd: float = 1.0, *, generator=None):
    global _warned
    if generator is not None:
        try:
            return torch.ops.aten.exponential_(
                self, lambd=lambd, generator=generator
            )
        except TypeError:
            if not _warned:
                _warned = True
                import logging

                logging.getLogger(__name__).warning(
                    "exponential_ on this stack ignores the generator "
                    "argument (vendor kernel lacks support); per-request "
                    "seeded sampling falls back to the global RNG."
                )
    return torch.ops.aten.exponential_(self, lambd=lambd)


def apply_exponential_compat() -> bool:
    """Install the generator-aware exponential_ shim.

    Returns True if the shim was installed in this call.
    """
    global _installed
    if _installed:
        return False
    torch.Tensor.exponential_ = _exponential_compat
    _installed = True
    return True
