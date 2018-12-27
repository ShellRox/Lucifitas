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
from __future__ import division  # Python 2
from errors import TreeError, ManagerError, QueryError
import numpy as np
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
        self.node_array = {}

    def set_node_array(self, node_array):
        if isinstance(node_array, dict):
            if all([isinstance(node[0], InternalNode) or isinstance(node[0], RootNode) for node in node_array.values()]):
                if not isinstance(node_array.get(0, [InternalNode])[0], RootNode):
                    raise TreeError("node_array must contain RootNode as its first element")
                else:
                    self.root_node = node_array[0]
                    self.node_array = node_array
            else:
                raise TreeError("node_array dictionary objects shall only contain InternalNode and RootNode values")
        else:
            raise TreeError("node_array must be a dict object")

    def add_node(self, tree_node):  # Negligible for current version, but might be helpful in the future
        if isinstance(tree_node, RootNode):
            if not self.root_node:  # Root node must obviously be the first node
                self.root_node = tree_node
                self.node_array.__setitem__(0, self.root_node)
            else:
                self.node_array.__delitem__(0)
                self.root_node = tree_node
                self.node_array.__setitem__(0, self.root_node)
        elif isinstance(tree_node, InternalNode):
            if not self.root_node:
                raise TreeError("RootNode instance must be added first")
            else:
                self.node_array.__setitem__(len(self.node_array) - 1, tree_node)
        else:
            raise TreeError("input must be InternalNode or RootNode instance")

    def remove_node(self, tree_node):  # Negligible for current version, but might be helpful in the future
        if tree_node in self.node_array.values():
            self.node_array.__delitem__(self.node_array.values().index(tree_node))
        else:
            raise TreeError("input must be present in node_array")

    def list_all(self):
        all_nodes = []
        for node in self.node_array.values():
            all_nodes.extend(node)
        return all_nodes

    def show_root(self):
        if not self.root_node:
            raise TreeError("root node not present in tree")
        else:
            return self.root_node

    def show_depth(self):  # maximum depth of partition tree
        return len(self.node_array)

    def list_internal(self):
        return [node for node in self.list_all() if isinstance(node, InternalNode)]

    def list_leaves(self):
        return [node for node in self.list_internal() if node.is_leaf()]

    def total_nodes(self):
        return len(self.list_all())

    def left_child_percentage(self):  # TODO: fix
        left_nodes = [node for node in self.list_internal() if node.order == -1 and not node.is_leaf()]
        return (len(left_nodes)/(self.total_nodes() - 1)) * 100

    def right_child_percentage(self):
        right_nodes = [node for node in self.list_internal() if node.order == 1 and not node.is_leaf()]
        return (len(right_nodes)/(self.total_nodes() - 1)) * 100

    def leaf_node_percentage(self):
        leaf_nodes = self.list_leaves()
        return (len(leaf_nodes))/(self.total_nodes() - 1) * 100


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
        self.children = ()
        self.order = 0
        self.percentile = 0
        # self.check_variables()

    def show_space(self):
        return self.space

    def list_points(self):  # Linker of self.show_space
        return self.show_space()

    def cell_count(self):
        return len(self.space)

    def check_variables(self):
        if not isinstance(self.space, np.ndarray):
            raise TreeError("vector space must be in the form of Numpy array")

    def add_children(self, left, right):
        self.children = (left, right)

    def set_percentile(self):  # TODO: make this function even faster
        scale_values = [np.inner(self.random_coefficients(), point[:self.indices].ravel()) for point in self.space]
        percentile = np.percentile(scale_values, self.ratio * 100)
        self.percentile = percentile
        return True

    @staticmethod
    def is_leaf():
        return False

    def random_test(self, main_point):
        percentile = self.ratio
        main_term = np.inner(main_point[:self.indices].ravel(), self.random_coefficients())
        if (main_term - percentile) >= 0:  # Hyper-plane equation defined in the document
            return -1  # Next node is the left child
        else:
            return 1  # Next node is the right child

    def random_coefficients(self):
        return np.random.uniform(size=self.indices)


class InternalNode(object):
    def __init__(self, capacity, ratio, indices, parent, order):
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
        self.parent = parent
        self.children = ()
        self.percentile = 0
        self.order = order
        self.check_variables()

    def add_point(self, point):
        self.points.append(point)
        # self.check_variables()

    def add_children(self, left, right):
        self.children = (left, right)

    def list_points(self):
        return self.points

    def is_leaf(self):
        return self.cell_count() <= self.capacity

    def cell_count(self):
        return len(self.points)

    def random_test(self, main_point):
        percentile = self.percentile
        main_term = np.inner(main_point[:self.indices].ravel(), self.random_coefficients())
        if (main_term - percentile) >= 0:  # Hyper-plane equation defined in the document
            return -1  # Next node is the left child
        else:
            return 1  # Next node is the right child

    def set_percentile(self):
        scale_values = [np.inner(self.random_coefficients(), point[:self.indices].ravel()) for point in self.points]
        percentile = np.percentile(scale_values, self.ratio * 100)
        self.percentile = percentile
        return True

    def random_coefficients(self):
        return np.random.uniform(size=self.indices)

    def check_variables(self):
        if not all([Manager.is_digit(i) for i in [self.ratio, self.capacity, self.indices]]):
            return ManagerError("split_ratio, capacity, indices must be all digits")
        if any([point for point in self.points if not isinstance(point, np.ndarray)]):
            return ManagerError("every point in node must be in the form of Numpy array")
        else:
            if self.cell_count() > 0:
                vec_dim = [vec.ndim for vec in self.points]  # Identifying whether or not passed data is a vector space
                if vec_dim.count(vec_dim[0]) != len(vec_dim):
                    return ManagerError("every basis of vector space must be equal")
        if not isinstance(self.parent, InternalNode) or isinstance(self.parent, RootNode):
            return TreeError("parent node must be either InternalNode or RootNode instance")


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
        """
        self.vector_space = vector_space
        self.tree_count = 30
        self.split_ratio = 1/2
        self.capacity = 12
        self.indices = 2
        self.random_forest = []

    def create_forest(self):
        for _ in range(0, self.tree_count):
            self.index_space()
        return self.random_forest

    def index_space(self):  # TODO: Improve tree creation performance even more
        shuffled_space = self.shuffle_space()
        current_tree = PartitionTree()
        level = 0  # Depth of the tree
        root_node = RootNode(shuffled_space, self.capacity, self.split_ratio, self.indices)
        node_array = {0: [root_node]}  # Dictionary containing array of every object at every level of the tree
        while True:
            current_nodes = node_array[level]
            if all([node.is_leaf() for node in current_nodes]):  # If we hit the depth
                break
            else:
                level += 1
                node_array[level] = []  # Empty array to be filled at each level
                for current_node in current_nodes:
                    if not current_node.is_leaf():
                        current_node.set_percentile()  # Bias that modifies hyper-plane according to split ratio
                        left_child = InternalNode(self.capacity, self.split_ratio, self.indices, current_node, -1)
                        right_child = InternalNode(self.capacity, self.split_ratio, self.indices, current_node, 1)
                        for point in current_node.list_points():
                            if current_node.random_test(point) == 1:  # If node should be a right child
                                right_child.add_point(point)
                            else:
                                left_child.add_point(point)
                        current_node.add_children(left_child, right_child)
                        node_array[level].extend([left_child, right_child])  # Add children to the next level
        current_tree.set_node_array(node_array)
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
    def save_forest(forest_array, *file_path):  # TODO: Implement more efficient method for saving large forest
        if all([isinstance(tree, PartitionTree) for tree in forest_array]):
            if not file_path:
                current_path = os.path.dirname(os.path.abspath(__file__))
                directory_path = os.path.join(current_path, "index_data")
                os.mkdir(directory_path)  # Directory must be created first
                file_path = os.path.join(directory_path, "{0}.p".format(time.time()))
            else:
                if not os.path.splitext(file_path)[1] == ".p":
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
            return []
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

    @staticmethod
    def _append(_list, _item_list):
        for _item in _item_list:
            _list.append(_item)
        return _list


class Query(object):
    """
    Input variables:
    random_forest - Array full of PartitionTree objects
    """
    def __init__(self):
        self.partition_forest = []

    def import_forest(self, forest):
        if self.verify_forest(forest):
            self.partition_forest = forest
        else:
            raise QueryError("partition forest must contain PartitionTree objects only")

    def search(self, query_vector):
        if isinstance(query_vector, np.ndarray):
            results = []
            for tree in self.partition_forest:
                current_node = tree.root_node[0]
                while not current_node.is_leaf():
                    if current_node.random_test(query_vector) == 1:
                        current_node = current_node.children[1]  # Right child
                    else:
                        current_node = current_node.children[0]  # Left child
                for points in current_node.list_points():
                    results.append(tuple(points.ravel().tolist()))  # TODO: improve speed by pre-setting
            return set().union(results)  # TODO: find a faster way (maybe?)
        else:
            raise QueryError("query vector must be a numpy array")

    @staticmethod
    def get_root(partition_forest):
        return partition_forest.root_node

    @staticmethod
    def verify_forest(forest):
        if all([isinstance(tree, PartitionTree) for tree in forest]):
            return True
        else:
            return False

    @staticmethod
    def custom_union(results, node):
        for point in node.list_points():
            if point not in results:
                results.append(point)
        return results

