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
from core.semantic.polyconvex import Manager, Query
from core.semantic.sequential import Sequential
import time


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
        tm = time.time()
        manager.create_forest()
        print("\nIndexing partition forest took {0}s ({1}, {2}, {3}, {4}, {5})\n".format(round(time.time() - tm, 2),
                                                                                         vector_space.size,
                                                                                         manager.tree_count,
                                                                                         manager.split_ratio,
                                                                                         manager.capacity,
                                                                                         manager.indices))
        self.assertTrue(len(manager.random_forest), manager.tree_count)  # Forest must be the size of tree_count

    def test_query(self):
        vector_space = np.asarray([np.random.rand(512, 1) for _ in range(0, 500)])
        query_vector = np.random.rand(512, 1)
        manager = Manager(vector_space)
        manager.create_forest()
        polyhedral_query = Query()
        polyhedral_query.import_data_by_array(manager.random_forest)
        sequential_query = Sequential(vector_space)
        sequential_results = sequential_query.query(query_vector)[0]
        polyhedral_results = polyhedral_query.query(query_vector)
        self.assertTrue(sequential_results[0] in polyhedral_results)


if __name__ == '__main__':
    unittest.main()
