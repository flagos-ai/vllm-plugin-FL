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

"""W2b model_runner_tail test package.

The import shim must run before any test module imports ``vllm`` (it
installs local-only placeholders and symbol backfills the CI image does
not need). A package's ``__init__`` executes before any of its modules,
so the shim is installed here; the ``from . import _imports`` line in
each test file is then an idempotent re-assertion that also satisfies
ruff's import ordering.
"""

from . import _imports  # noqa: F401  (installs local import shims first)
