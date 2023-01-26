from alghoritms import *
from queue import Queue
# s = Dsu()
# print(s.storage)
# print('')
# print(s.find_set(0))
# count = 0
# for i in s.storage.values():
#     if i == 3:
#         count += 1
# print('')
# print(count)
# print('')
# print(s.storage)
# from queue import PriorityQueue

# q = PriorityQueue()
        # {0: 1, 1: 4, 2: 3, 3: 3, 4: 3, 5: 0}
# m = {1}
# print(type(m))

# lst_of_weightedges = [
#     ('a', 'b', 1), ('a', 'x', 3), ('a', 'y', 4), ('b', 'a', 1), ('b', 'c', 1), ('b', 'd', 2),
#     ('b', 'e', 8), ('c', 'b', 1), ('c', 'e', 20), ('d', 'b', 2), ('d', 'e', 5), ('e', 'b', 8),
#     ('e', 'c', 20), ('e', 'd', 5), ('x', 'a', 3), ('x', 'y', 0), ('y', 'a', 4), ('y', 'x', 0)
# ]
# print(sorted(lst_of_weightedges, key=lambda x: x[2]))

# s = Dsu()
# a = 1
# if not s.have_element(a):
#     s.make_set(a)
# print(not s.have_element(a))

# lst_of_weightedges = [
#     ('a', 'b', 1), ('a', 'x', 3), ('b', 'c', 1), ('b', 'd', 2),
#     ('c', 'b', 1), ('c', 'e', 20), ('d', 'b', 2), ('d', 'e', 5), ('e', 'b', 8),
#     ('e', 'c', 20), ('e', 'd', 5), ('x', 'a', 3), ('x', 'y', 0), ('y', 'a', 4), ('y', 'x', 0)
# ]
# print(prims_algorithm_naive(lst_of_weightedges))
# a = get_weightgraph_listadjacency(lst_of_weightedges, numbers=False)
# print(prims_algorithm_optimized_log(a))
# print(kruskals_algorithm(lst_of_weightedges))
# edges = [(a, b) for (a, b, w) in lst_of_weightedges]
# a = get_graph_listadjacency(edges, numbers=False)
# print(graph_bfs(a, 'b')[1])
# print(type(1.8446744e+19))

# (a, b, 1) +
# (b, c, 1) +
# (b, d, 2) +
# (a, x, 3) +
# (x, y, 0) +
# (d, e, 5) +



# a = get_weightgraph_listadjacency(lst_of_weightedges, numbers=False)
# print(prims_algorithm_naive(lst_of_weightedges))
# print(prims_algorithm_optimized_square(a))
# print(prims_algorithm_optimized_log(a))

# U = {1}
# print(len(U))



# print(q.__doc__)
# q = PriorityQueue()
# edge = ('a', 'y', 4)
# w = edge[2]
# q.put((w, edge))
# edge = ('a', 'b', 1)
# w = edge[2]
# # print(q.queue)
# q.put((w, edge))
# print(q.queue, (w, edge))
# print((w, edge) in q.queue)
# print(q.get())
# print(q.get())



# q.put((1, 3))
# q.put((2, 4))
# q.put((0, 88, 3))
# print(bool(q.qsize()))
# print(q.get())
# print(q.get())





# lst_of_weightedges = [
#     ('a', 'b', 1), ('a', 'x', 3), ('a', 'y', 4), ('b', 'a', 1), ('b', 'c', 1), ('b', 'd', 2),
#     ('b', 'e', 8), ('c', 'b', 1), ('c', 'e', 20), ('d', 'b', 2), ('d', 'e', 5), ('e', 'b', 8),
#     ('e', 'c', 20), ('e', 'd', 5), ('x', 'a', 3), ('x', 'y', 0), ('y', 'a', 4), ('y', 'x', 0)
# ]
# print(prims_algorithm_naive(lst_of_weightedges))
# lst = get_weightgraph_listadjacency(lst_of_weightedges, numbers=False)
# print(prims_algorithm_optimized_square(lst))



# lst_of_edges1 = [('m', 'a'), ('a', 'b'), ('b', 'c'), ('b', 'd'), ('c', 'e'), ('d', 'e'), ('e', 'b'), ('x', 'y'), ('y', 'x')]
# lst_of_edges2 = [(0, 1), (1, 0), (1, 2), (1, 3), (1, 4), (2, 1), (2, 4), (3, 1), (3, 4), (4, 1), (4, 2), (4, 3), (4, 5), (5, 4), (5, 6), (6, 5), (7, 8), (8, 7)]
# a = get_graph_listadjacency(lst_of_edges1, numbers=False)
# b = get_graph_listadjacency(lst_of_edges2)
# print(have_cycle(a))
# print(topological_sort(a))
# print(topological_sort(b))
# lst_of_weightedges1 = [('a', 'b', 1), ('b', 'a', 2), ('b', 'c', 1), ('b', 'd', 2), ('c', 'e', 20), ('d', 'e', 5), ('e', 'b', 8), ('x', 'y', 0), ('y', 'x', 0)]
# lst_of_weightedges2 = [
#     (0, 1, 1), (1, 0, 1), (1, 2, 1), (1, 3, 1),
#     (1, 4, 1), (2, 1, 1), (2, 4, 1), (3, 1, 1),
#     (3, 4, 1), (4, 1, 1), (4, 2, 1), (4, 3, 1),
#     (4, 5, 1), (5, 4, 1), (5, 6, 1), (6, 5, 1)
#     ]

# a = []
# a.append(get_graph_listadjacency(lst_of_edges1, numbers=False))
# a.append(get_graph_listadjacency(lst_of_edges2))
# a.append(get_weightgraph_listadjacency(lst_of_weightedges1, numbers=False))
# a.append(get_weightgraph_listadjacency(lst_of_weightedges2))

# for i in range(4):
#     if i > 1:
#         print(is_bipartite(a[i], weight=True))
#     else:
#         print(is_bipartite(a[i]))


# lst_of_weightedges2 = [
#     (0, 1, 1), (1, 0, 1), (1, 2, 1), (1, 3, 1),
#     (1, 4, 1), (2, 1, 1), (2, 4, 1), (3, 1, 1),
#     (3, 4, 1), (4, 1, 1), (4, 2, 1), (4, 3, 1),
#     (4, 5, 1), (5, 4, 1), (5, 6, 1), (6, 5, 1)
# ]



# lst_of_edges = [
#     (0, 1), (1, 0), (1, 2),
#     (1, 5), (2, 1), (2, 3),
#     (2, 4), (3, 2), (3, 5),
#     (4, 2), (4, 5), (5, 1),
#     (5, 3), (5, 4)
# ]


# a = get_graph_listadjacency(lst_of_edges)
# print(is_bipartite(a))


# class SomeException(Exception):

#     def __init__(self):
#         # self.messages = string
#         super().__init__()

# def Somefunction(a:int, s:str):
#     if a < 7:
#         a += 1
#         s += str(a) + "+"
#         try:
#             s = Somefunction(a, s)
#             return s
#         except SomeException:
#             return s
#     else:
#         raise SomeException

# print(Somefunction(1, ''))


# lst_of_edges = [
#     (0, 1), (1, 2), (1, 3), (2, 0), (3, 1)
# ]
# lst_of_edges = [
#     (1, 2), (1, 3),
#     (2, 1), (2, 3),
#     (3, 1), (3, 2)
# ]
# lst_of_edges = [
#     ('a', 'b'), ('a', 'c'),
#     ('b', 'a'), ('b', 'c'),
#     ('c', 'a'), ('c', 'b')
# ]
# a = get_graph_listadjacency(lst_of_edges, numbers=False)
# print(a)
# print(have_cycle(a))


# from queue import PriorityQueue
# q = PriorityQueue()

# q.put((1, 2))
# (a, b) = q.get()
# print(a)
# print(b)

# d = int(1.8446744e+19)
# print(type(d))
# q = PriorityQueue()
# q.put((2, 'A'))
# q.put((1, 'B'))
# print(q.get())


# lst_of_weightedges1 = [
#     (0, 1, 1), (1, 0, 1), (1, 2, 1), (1, 3, 1),
#     (1, 4, 1), (2, 1, 1), (2, 4, 1), (3, 1, 1),
#     (3, 4, 1), (4, 1, 1), (4, 2, 1), (4, 3, 1),
#     (4, 5, 1), (5, 4, 1), (5, 6, 1), (6, 5, 1)
#     ]
# lst_of_weightedges2 = [('a', 'b', 1), ('b', 'a', 2), ('b', 'c', 1), ('b', 'd', 2), ('c', 'e', 20), ('d', 'e', 5), ('e', 'b', 8), ('x', 'y', 0), ('y', 'x', 0)]
# adj1 = get_weightgraph_listadjacency(lst_of_weightedges1)
# adj2 = get_weightgraph_listadjacency(lst_of_weightedges2 ,numbers=False)
# print(dejcstra_heap(adj1, 1))
# print(dejcstra_heap(adj2, 'c'))

# lst_of_edges = [(0, 1), (1, 0), (1, 2), (1, 3), (1, 4), (2, 1), (2, 4), (3, 1), (3, 4), (4, 1), (4, 2), (4, 3), (4, 5), (5, 4), (5, 6), (6, 5)]
# adj = get_weightgraph_listadjacency(lst_of_weightedges, numbers=False)
# print(adj)
# print(len(adj))
# print(graph_bfs(adj, 'a', weight=True))



# from queue import Queue
# q = Queue()
# q.get() ---- pop()
# q.put(element) --- push(element)
# q.qsize() --- len(q)
# q.empty() --- True when q is empty, otherwise False
# print(dir(q))
# print(' ')
# print(q.empty())
# q.put('a')
# print(q.empty())
# q.put(1)
# print(repr(q))
# print(q.get())
# print(q.qsize())
# print(q.get())
# print(q.qsize())




# lst_of_weightedges = [('a', 'b', 1), ('b', 'a', 2), ('b', 'c', 1), ('b', 'd', 2), ('c', 'e', 2), ('d', 'e', 5), ('e', 'b', 8)]
# lst_of_edges = [('a', 'b'), ('b', 'a'), ('b', 'c'), ('b', 'd'), ('c', 'e'), ('d', 'e'), ('e', 'b'), ('x', 'y'), ('y', 'z'), ('z', 'v'), ('v', 'x')]
# adj = get_graph_listadjacency(lst_of_edges, numbers=False)
# color = get_empty_color(adj)
# graph_dfs(adj, color, 'a')
# print(color)
# lst_of_edges = [(0, 1), (1, 0), (1, 2), (1, 3), (2, 1), (2, 4), (3, 1), (3, 4), (4, 1), (4, 2), (4, 3)]
# lst_adjacency = get_graph_listadjacency(lst_of_edges, numbers=False)
# color = [0 for i in range(5)]
# print(lst_adjacency)
# print(graph_dodfs(lst_adjacency))
# u = 1
# graph_dfs(lst_adjacency, color, u)
# print(graph_dodfs(lst_adjacency))
# matrix = get_weightgraph_matrixadjacency(lst_of_edges)
# for i in matrix:
#     print(i)





# lst_of_adjancency = get_weightgraph_listadjacency(lst_of_edges)
# print(lst_of_adjancency)
# print('')
# element = lst_of_adjancency[1]
# print(element)
# f = +1.875752990E+00
# print(f)
# a = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 1)]
# n = 5
# lst = [[]] * (n + 1)
# for element in a:
#     vertex_a = element[0]
#     vertex_b = element[1]
#     lst[vertex_a].append(vertex_b)
# print(lst)

# lst = [[]] * 4
# lst[1].append(3)
# print(lst)

# lst = [0] * 6
# for i in range(len(lst)):
#     print(id(lst[i]))
# lst[3] = 3
# print(' ')
# for i in range(len(lst)):
#     print(id(lst[i]))


# lst = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 1)]
# n = 0
# for (a, b) in lst:
#     mx = max(a, b)
#     if mx > n:
#         n = mx
# adjlst = [[] for i in range(n + 1)]
# for (a, b) in lst:
#     adjlst[a].append(b)
# print(adjlst)

# s = 'babababab'
# sub = 'bab'
# print(strFind_KMP(sub, s))
# print(strFind_KMP_bonus(sub, s))
# lst = [[], [], []]
# lst[0].append(2)
# lst[0].append(2)
# lst[0].append(2)
# lst[0].append(2)
# lst[0].append(2)
# print(lst)
# d = Hash_table()
# d.insert(('abc', 12))
# d.insert(('bm', 1))
# d.insert(('gm', 2))
# print(d)
# print(d['abc'])
# d['abc'] = 10
# print(d['abc'])
# print(d.table)
# d = Hash_table()
# d['hello'] = 'World'
# d[24] = 54
# print(d)
# print(d['hello'])
# print('')
# d['hello'] = 'Nikita'
# print(d)
# print(d['hello'])
# print(dir(d))

# try:
#     a = []
#     print(a[-1])
# except IndexError:
#     print('Nononono ', a)

# lst_of_edges = [(0, 1), (1, 0), (1, 2), (1, 3), (2, 4), (3, 4), (4, 1)]
# d = {a: {a: 0 for (a, b) in lst_of_edges} for (a, b) in lst_of_edges}
# for (a, b) in lst_of_edges:
#     d[a] = {a: 0 for (a, b) in lst_of_edges}
# for i in d.values():
#     print(i)


# lst_of_adjancency = get_graph_listadjacency(lst_of_edges)
# matrix = get_graph_matrixadjacency(lst_of_edges, numbers=False)
# print(lst_of_adjancency)
# print(' ')
# for i in matrix.values():
#     print(i)
# print(d)

# d = {}
# d['a'] = 123
# print(d)
# b = ()
# print(isinstance(b, tuple))
# print('a' != b)
# d = Hash_table()
# d.insert(('abc', 12))
# d.insert(('bm', 1))
# d.insert(('gm', 2))
# d.insert(('vm', 3))
# d.insert(('cm', 4))
# d.insert(('lm', 5))
# print(d)
# print(d.n)
# print(d.m)
# print(d.alpha)
# d.insert(('wm', 6))
# d.insert(('om', 7))
# d.insert(('mm', 8))
# print(d)
# print(d.n)
# print(d.m)
# print(d.alpha)
# print(' ')
# print(d.get_element('wm'))
# d.delete('wm')
# print(d)
# print(d.get_element('ym'))
# print("Can't to delete a non-existent key")
# s = 'asdjkbdsf'
# print(strHash(s))
# from random import randint
# n = 0
# lst = [None] * n
# print(lst)
# p = 3
# def some_function(key):
#     if isinstance(key, float):
#         key = int(key)
#     return key + 1
# a = 1.1
# print(some_function(a))
# print(a)
# print(randint(1, p - 1))
# print(modular_exponentiation(7, 560, 561))
# print(go_to_binary(10))
# print(bin(10)[2:])
# print(modular_linear_solver(14, 30, 100))
# print(gcd_euclid(99, 78))
# print(euclid_extended(99, 78))
# print(pow(9, 9))
# a = 4
# n = 3
# print(pow(a, n))
# a = [2,1,5,3,9,10,4,5,3,2,1,5,8,23,7,0,54,21,43,12,23,1,2,3,4,9,5,2,1,5,2,0,32,54,1,45,78,12,32,93,76,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,11,1,1,11,1,1,1,11,1,11,1,1,11,1,11,1,11,1,11,11,1]
# b = [2,1,5,3,9,10,3,5,4]
# c = bubleSort(b, reverse=True)
# print(c)
# pattern = "pa"
# text = "parapapa"
# text1 = "parapapaparapraspaarpapara"
# print(strFind_naive(pattern, text))
# print(strFind_naive(pattern, text1))
# print("Hi")
# a = Poly(1,2,3,4)
# print(a)
# a = [1,2,3]
# print(somefunc(a, 1))
# print(horners_rule(a, 0))
# print(horners_rule(a, 1))
# print(horners_rule(a, 2))
# txt = "GEEKS FOR GEEKS"
# pat = "GEEK"
# pattern = "pa"
# text = "parapapa"
# text1 = "parapapaparapraspaarpapara"
# print(strFind_naive(pattern, text))
# print(strFind_naive(pattern, text1))
# print(" ")
# print(strFind_RabinKarp(pattern, text))
# print(strFind_RabinKarp(pattern, text1))

# def numstringToint(string):
#     n = len(string)
#     sum = 0
#     for i in range(n):
#         sum = int(string[i]) + 10*sum
#     return sum

# def p(string, d=256, q=101):
#     n = len(string)
#     sum = 0
#     for i in range(n):
#         sum = (ord(string[i]) + d*sum)
#     return sum % q

# def p1(string, d=256, q=101):
#     n = len(string)
#     sum = 0
#     for i in range(n):
#         sum = (ord(string[i]) + d*sum) % q
#     return sum

# string = 'asdjkhg'
# print(p(string))
# print(p1(string))

# string = "12345"
# value = numstringToint(string)
# print(value)