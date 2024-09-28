
from tasks cimport Tasks, TaskIDList

cdef extern from "include/graph.hpp":

    cdef cppclass GraphManager:
        @staticmethod
        void populate_dependents(Tasks &tasks)
        @staticmethod
        TaskIDList random_topological_sort(Tasks &tasks)

