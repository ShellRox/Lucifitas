"""
MIT License

Copyright (c) 2018 ShellRox

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Unit tests for core/semantic/polyconvex.py algorithm
"""
import unittest
import numpy as np
from core.semantic.polyconvex import Manager


class TestPartitionManager(unittest.TestCase):
    def test_manager(self):  # Test if Manager can be initiated
        vector_space = [np.random.rand(512, 1) for _ in range(0, 100)]
        self.assertTrue(Manager(vector_space))

    def test_random_tests(self):  # Test if single tree indexing works
        vector_space = np.asarray([np.random.rand(512, 1) for _ in range(0, 100)])
        manager = Manager(vector_space)
        prepared_tree = manager.index_space()
        self.assertFalse(any([i.capacity > manager.capacity  # No leaf node shall exceed capacity
                              for i in prepared_tree.list_leaves()]))

    def test_random_forest(self):
        vector_space = np.asarray([np.random.rand(512, 1) for _ in range(0, 100)])
        manager = Manager(vector_space)
        manager.create_forest()
        self.assertTrue(len(manager.random_forest), manager.tree_count)  # Forest must be the size of tree_count


if __name__ == '__main__':
    unittest.main()
