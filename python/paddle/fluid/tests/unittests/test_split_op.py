#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.profiler as profiler
import os

SIZE_X = int(os.getenv("SIZE_X"))
SIZE_Y = int(os.getenv("SIZE_Y"))
SIZE_Z = int(os.getenv("SIZE_Z"))
AXIS = int(os.getenv("AXIS"))


class TestSplitOp(OpTest):
    def setUp(self):
        self._set_op_type()

        axis = AXIS

        size_x = SIZE_X
        size_y = SIZE_Y
        size_z = SIZE_Z

        print(size_x, size_y, size_z, axis)

        if axis == 2:
            x = np.random.random((size_z, size_y, size_x * 3)).astype('float32')
            out = np.split(x, [size_x, 2 * size_x], axis)
            self.attrs = {'axis': axis, 'sections': [size_x, size_x, size_x]}
        elif axis == 0:
            x = np.random.random((size_z * 3, size_y)).astype('float32')
            out = np.split(x, [size_z, 2 * size_z], axis)
            self.attrs = {'axis': axis, 'sections': [size_z, size_z, size_z]}
        self.inputs = {'X': x}
        self.outputs = {'Out': [('out%d' % i, out[i]) \
            for i in range(len(out))]}

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output()
        with profiler.profiler("All", 'total', "split_profiler.log") as prof:
            for iter in range(100):
                self.check_output()
        #print(self.inputs)
        #print(self.outputs)

        #def test_check_grad(self):
        #    self.check_grad(['X'], ['out0', 'out1', 'out2'])

        #class TestSplitByrefOp(OpTest):
        #    def _set_op_type(self):
        #        self.op_type = "split_byref"


if __name__ == '__main__':
    unittest.main()
