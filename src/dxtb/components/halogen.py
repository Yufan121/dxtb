# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
"""
Components: Halogen
===================

Tight-binding components for halogen bond corrections.
"""

from dxtb._src.components.classicals.halogen import Halogen as Halogen
from dxtb._src.components.classicals.halogen import new_halogen as new_halogen

__all__ = [
    "Halogen",
    "new_halogen",
]