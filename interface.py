from task4feedback.fastsim.simulator import Simulator


s = Simulator(5)
s.add_task(0, [])
s.add_task(1, [0])
s.add_task(2, [1])
s.add_task(3, [2])
s.add_task(4, [3])

s.initialize_dependents()

s.print_task(0)

arr = s.random_topological_sort()

print(arr)
