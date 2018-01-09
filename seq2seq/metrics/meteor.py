# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BLEU metric implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import subprocess
import pkg_resources
import tempfile
import numpy as np

METEOR_JAR = 'meteor/meteor-1.5.jar'




def compute(refs, hyps, language="de", norm=False):
  cmdline = ["java", "-Xmx2G", "-jar", METEOR_JAR]

  if isinstance(hyps, list):
      hypothesis_file = tempfile.NamedTemporaryFile()
      hypothesis_file.write("\n".join(hyps).encode("utf-8"))
      hypothesis_file.write(b"\n")
      hypothesis_file.flush()


  cmdline.append(hypothesis_file.name)

  if isinstance(refs, list):
      reference_file = tempfile.NamedTemporaryFile()
      reference_file.write("\n".join(refs).encode("utf-8"))
      reference_file.write(b"\n")
      reference_file.flush()

  cmdline.append(reference_file.name)


  cmdline.extend(["-l", "de"])
  if norm:
    cmdline.append("-norm")

  score = subprocess.run(cmdline, stdout=subprocess.PIPE,
                         universal_newlines=True).stdout.splitlines()
  if len(score) == 0:
    return np.float32(0.0)
  else:
    # Final score: 0.320320320320
    return np.float32(score[-1].split(":")[-1].strip())