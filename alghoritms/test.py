from alghoritms import *
# d = Hash_table()
# d.insert(('abc', 12))
# d.insert(('bm', 1))
# d.insert(('gm', 2))
# print(d)
# print(d['abc'])
# d['abc'] = 10
# print(d['abc'])
# print(d.table)


try:
    a = []
    print(a[-1])
except IndexError:
    print('Nononono ', a)




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