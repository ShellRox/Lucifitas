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

https://arxiv.org/ftp/arxiv/papers/1505/1505.03090.pdf

Algorithm for dividing high-dimensional vector space into non-overlapping cells
of hyper convex-polyhedrons bounded by hyper-planes using random binary partitioning
technique.
"""
from errors import TreeError, ManagerError
import numpy as np
import random
import os
import pickle
import time


class PartitionTree(object):
    def __init__(self):
        """
        Pre-defined variables:
        node_array - array containing all types of nodes (root, internal and leaf respectively)
        """
        self.root_node = None
        self.node_array = []

    def add_node(self, tree_node):
        if isinstance(tree_node, RootNode):
            if not self.root_node:
                self.root_node = tree_node
                self.node_array.append(self.root_node)
            else:
                self.node_array.remove(self.root_node)
                self.root_node = tree_node
                self.node_array.append(self.root_node)
        elif isinstance(tree_node, InternalNode):
            if not self.root_node:
                raise TreeError("RootNode instance must be added first")
            else:
                self.node_array.append(tree_node)
        else:
            raise TreeError("input must be InternalNode or RootNode instance")

    def remove_node(self, tree_node):
        if tree_node in self.node_array:
            self.node_array.remove(tree_node)
        else:
            raise TreeError("input must be present in node_array")

    def list_all(self):
        return self.node_array

    def show_root(self):
        if not self.root_node:
            raise TreeError("root node not present in tree")
        else:
            return self.root_node

    def list_internal(self):
        return [node for node in self.node_array if isinstance(node, InternalNode)]

    def list_leaves(self):
        return [node for node in self.list_internal() if node.is_leaf()]


class RootNode(object):
    def __init__(self, vector_space, capacity, ratio, indices):
        """
        Input variables:
        vector_space - n-dimensional vector space in the form of Numpy array
        split_ratio - balance variable of a tree from least (0) to most (1/2)
        capacity - the maximum number of database points in a leaf node
        indices - the number of indices used in the subspace projection in the random tests
        """
        self.space = vector_space
        self.capacity = capacity
        self.ratio = ratio
        self.indices = indices
        self.node_position = [0]
        self.check_variables()

    def show_space(self):
        return self.space

    def cell_count(self):
        return len(self.space)

    def check_variables(self):
        if not isinstance(self.space, np.ndarray):
            raise TreeError("vector space must be in the form of Numpy array")

    @staticmethod
    def is_leaf():
        return False

    def random_test(self, main_point):
        # random_coefficients = self.random_coefficients()
        scale_values = np.array(sorted([np.inner(self.random_coefficients(), point[:self.indices].ravel())
                                        for point in self.space]))
        percentile = random.choice([np.percentile(scale_values, 100 * self.ratio),
                                    np.percentile(scale_values, 100 * (1 - self.ratio))])
        main_term = np.inner(main_point[:self.indices].ravel(), self.random_coefficients())
        if (main_term - percentile) >= 0:  # Hyper-plane equation defined in the document
            return -1  # Next node is the left child
        else:
            return 1  # Next node is the right child

    def random_coefficients(self):
        return np.random.randint(2, size=self.indices)


class InternalNode(object):
    def __init__(self, capacity, ratio, indices, position):
        """
        Input variables:
        split_ratio - balance variable of a tree from least (0) to most (1/2)
        capacity - the maximum number of database points in a leaf node
        indices - the number of indices used in the subspace projection in the random tests
        position - previous coordinates of node in tree, where -1, 0, 1 = left, center and right respectively
        """
        self.points = []
        self.capacity = capacity
        self.ratio = ratio
        self.indices = indices
        self.node_position = position
        self.check_variables()

    def add_point(self, point):
        self.points.append(point)
        self.check_variables()

    def list_points(self):
        return self.points

    def is_leaf(self):
        return self.ratio * self.capacity < self.cell_count() <= self.capacity

    def cell_count(self):
        return len(self.points)

    def random_test(self, main_point):
        # random_coefficients = self.random_coefficients()
        scale_values = np.array(sorted([np.inner(self.random_coefficients(), point[:self.indices].ravel())
                                        for point in self.points]))
        percentile = random.choice([np.percentile(scale_values, 100 * self.ratio),  # Just as described on Section 3.1
                                    np.percentile(scale_values, 100 * (1 - self.ratio))])
        main_term = np.inner(main_point[:self.indices].ravel(), self.random_coefficients())
        if self.is_leaf():
            return 0  # Next node is the center leaf child
        else:
            if (main_term - percentile) >= 0:  # Hyper-plane equation defined in the document
                return -1  # Next node is the left child
            else:
                return 1  # Next node is the right child

    def random_coefficients(self):
        return np.random.randint(2, size=self.indices)

    def check_variables(self):
        if not all([Manager.is_digit(i) for i in [self.ratio, self.capacity, self.indices]]):
            return ManagerError("split_ratio, capacity, indices must be all digits")
        if any([point for point in self.points if not isinstance(point, np.ndarray)]):
            return ManagerError("every point in node must be in the form of Numpy array")
        else:
            if self.cell_count() > 0:
                vec_dim = [vec.ndim for vec in self.points]
                if vec_dim.count(vec_dim[0]) != len(vec_dim):
                    return ManagerError("every basis of vector space must be equal")
        if not isinstance(self.node_position, list) and len(self.node_position) == 0:
            return TreeError("previous position must be a list")


class Manager(object):
    def __init__(self, vector_space):
        """
        Input variables:
        vector_space - n-dimensional vector space in the form of Numpy array

        Pre-defined variables:
        tree_count - number of random partitions to use in the random forest
        split_ratio - balance variable of a tree from least (0) to most (1/2)
        capacity - the maximum number of database points in a leaf node
        indices - the number of indices used in the subspace projection in the random tests

        TODO: Create a query function for indexed data (coming soon)
        """
        self.vector_space = vector_space
        self.tree_count = 10
        self.split_ratio = 1/2
        self.capacity = 2
        self.indices = 1
        self.random_forest = []

    def create_forest(self):
        for _ in range(0, self.tree_count):
            self.index_space()
        return self.random_forest

    def index_space(self):
        shuffled_space = self.shuffle_space()
        current_tree = PartitionTree()
        root_node = RootNode(self.vector_space, self.capacity, self.split_ratio, self.indices)
        current_tree.node_array.append(root_node)
        position = list(root_node.node_position)  # Initial position
        position.append(root_node.random_test(shuffled_space[0]))
        for p in shuffled_space:  # Randomly pick feature vectors for partition
            while True:
                existent_node = self.node_exists(current_tree, position)
                if existent_node:  # If node already exists at certain position
                    current_node = existent_node
                else:
                    current_node = InternalNode(self.capacity, self.split_ratio, self.indices, position)
                current_node.add_point(p)
                current_tree.node_array.append(current_node)
                if not current_node.is_leaf():
                    position.append(current_node.random_test(p))
                else:
                    position = [root_node.node_position[0],
                                root_node.random_test(shuffled_space[0])]  # Reset position for next iteration
                    break
        self.random_forest.append(current_tree)
        return current_tree

    def shuffle_space(self):
        np.random.shuffle(self.vector_space)
        return self.vector_space

    def check_variables(self):
        if not all([self.is_digit(i) for i in [self.tree_count, self.split_ratio, self.capacity, self.indices]]):
            return ManagerError("tree_count, split_ratio, capacity, indices must be all digits")
        if not isinstance(self.vector_space, np.ndarray):
            return ManagerError("vector space must be in the form of Numpy array")
        else:
            if self.indices > self.vector_space.ndim:
                return ManagerError("number of indices should not be more than dimension of vector space")
        if not 0 < self.split_ratio <= 1/2:
            raise ManagerError("split_ratio should be greater than 0 and less than 1/2")

    @staticmethod
    def save_forest(forest_array, *file_path):
        if all([isinstance(tree, PartitionTree) for tree in forest_array]):
            if not file_path:
                current_path = os.path.dirname(os.path.abspath(__file__))
                directory_path = os.path.join(current_path, "index_data")
                os.mkdir(directory_path)
                file_path = os.path.join(directory_path, "{0}.p".format(time.time()))
            else:
                if os.path.splitext(file_path)[1] == ".p":
                    return ManagerError('file extension must be ".p"')
            with open(file_path, "w") as fl:
                fl.write(pickle.dumps(forest_array))
        else:
            return ManagerError("forest_array must be array full of PartitionTree objects")

    @staticmethod
    def node_exists(tree, position):
        if isinstance(tree, PartitionTree):
            for node in tree.node_array:
                if node.node_position == position:
                    return node
            return False
        else:
            return TreeError("tree must be passed as first argument")

    @staticmethod
    def is_digit(instance):
        try:
            int(instance)
        except ValueError:
            return False
        else:
            return True

