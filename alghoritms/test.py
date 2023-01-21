from alghoritms import *
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