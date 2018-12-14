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

Sequential nearest-neighbour query function which is suboptimal in practice
but consequentially useful to test more optimal query functions.
"""
import numpy as np
from errors import SequentialError


class Sequential(object):
    """
    Input variables:
    vector_space - n-dimensional vector space in the form of Numpy array
    """
    def __init__(self, vector_space):
        self.vector_space = vector_space

    def query(self, query_vector):
        most_similar = ([], 0)
        for v in self.vector_space:
            similarity = np.linalg.norm(query_vector.ravel()) - np.linalg.norm(v.ravel())
            if similarity > most_similar[1]:
                most_similar = (v, similarity)
        return most_similar

    def check_variables(self):
        if not isinstance(self.vector_space, np.ndarray):
            return SequentialError("vector space must be in the form of Numpy array")
