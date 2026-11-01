# Copyright 2023 The Project AETHER Authors.
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
"""Pytest configuration shared by the Project AETHER test suite.

The distributed tests need several XLA devices. On a CPU-only host, XLA can
expose the host as N virtual devices, but only if the flag is set before JAX
initializes its backends, so it is set here at collection time.
"""

import os

_NUM_VIRTUAL_DEVICES = 8

_flags = os.environ.get('XLA_FLAGS', '')
if 'xla_force_host_platform_device_count' not in _flags:
    os.environ['XLA_FLAGS'] = (
        f'{_flags} --xla_force_host_platform_device_count='
        f'{_NUM_VIRTUAL_DEVICES}').strip()
