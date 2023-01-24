import time
from sys import exit
from typing import Any, Iterable
from random import randint
from queue import Queue, PriorityQueue
#This library provides you with implementations of fundamental algorithms and data structures
#For each entity are presented Time and Memory complexity. If there is no inside entity words Time and Memory
#then inside must be word Complexity - it's mean Time complexity


#cs && searching algorithms
def go_to_binary(a:int) -> str:
    """
    This algorithm find reversed binary representation of number a

    Complexity: O(n) where n - count of digits in a

    Args:
        a (int): input number

    Returns:
        str: binary representation of a
    """
    s = ""
    while a != 0:
        s += str(a % 2)
        a //= 2
    return s


def linear_search(lst:list, a:Any) -> int:
    """
       This is naive algorithm of searching element a in collection lst
       This algorithm find for you index of element a in collection lst if element a is in the lst

    Complexity: O(n)

    Args:
        lst (list): any collection
        a (Any): some element

    Returns:
        int: index of element a in collection lst
    """
    for i in range(len(lst)):
        if lst[i] == a:
            return i


def binary_search(lst:list, x:int) -> int:
    """
       This is algorithm of binary searching in sorted by ascending collection.
       This algorithm find for you such index i that lst[i] == x is True
       Use this algorithm if you are sure that x exactly in lst and you need only index of x

    Complexity: O(log(n)) where n = len(lst)

    Args:
        lst (list): sorted by ascending collection
        x (int): some element

    Returns:
        int: index of element x in collection lst
    """
    l = 0
    r = len(lst) - 1
    while (r - l + 1) > 0:
        m = l + ((r - l) // 2)
        if lst[m] > x:
            r = m - 1
        elif lst[m] < x:
            l = m + 1
        else:
            return m
    return None


def binary_search_imaginary(lst:list, x:int) -> int:
    """
        This is algorithm of binary searching in sorted by ascending collection.
        This algorithm find for you such index i that x must be placed under that index.
        That algorithm can find index of existing element, and intended index of an element
        If algorithm returns -1 that mean that you need extend your collection and put x under index 0

    Complexity: O(log(n)) where n = len(lst)

    Args:
        lst (list): sorted by ascending collection
        x (int): some element

    Returns:
        int: index of element x in collection lst
    """
    l = -1
    r = len(lst)
    while r - l > 1:
        m = l + (r - l) // 2
        if lst[m] <= x:
            l = m
        else:
            r = m
    return l


#math
def pow(a:int , n:int) -> float:
    """ Возводить в степень можно гораздо быстрее,
        чем за n умножений! Для этого
        нужно воспользоваться следующими
        рекуррентными соотношениями:
        aⁿ = (a²)ⁿ/² при четном n,
        aⁿ=a⋅aⁿ⁻¹ при нечетном n.

    Complexity: O(logn)

    Args:
        a (int): value
        n (int): degree

    Returns:
        float: a ** n
    """
    if n % 2 == 0:
        return (a**2)**(n/2)
    if n % 2 != 0:
        return a*pow(a, n-1)
    return a


def horners_rule(a:list, x:int) -> int:
    """
       Horner's rule
       Calculation of the polynomial value at point x
       The polynomial is represented in this form:
       A(x) = a[0] + x * (a[1] + x * (a[2] + ... + x * (a[n-2] + x * (a[n-1]))...))
       This form allows you to efficiently calculate the value of a polynomial at a given point

    Complexity: θ(n)

    Args:
        a (list): iterable (may be list, tuple, ...)
        x (int): point

    Raises:
        ValueError: _description_

    Returns:
        int: value in point x
    """
    n = len(a)
    if n != 0:
        sum = 0
        for i in range(n-1, -1, -1):
            sum = a[i] + sum * x
        return sum
    else:
        raise ValueError


def gcd_euclid(a:int, b:int) -> int:
    """
    Euclid algorithm of finding greatest common divisor of numbers a and b

    Complexity: O(lg(min(a, b)))

    Args:
        a (int): some int
        b (int): some int

    Returns:
        int:
    """
    if b == 0:
        return a
    else:
        return gcd_euclid(b, a % b)


def euclid_extended(a:int, b:int) -> tuple:
    """
    Extended Euclid algorithm
    let d = gcd(a, b). d is linear combination a and b, so d = ax + by.
    This alhorithm returns tuple of the form (d, x, y)
    Use this algorithm if you need x and y

    Complexity: O(lg(min(a, b)))

    Args:
        a (int): some int
        b (int): some int

    Returns:
        tuple: (d, x, y)
    """
    if b == 0:
        return (a, 1, 0)
    else:
        v = euclid_extended(b, a % b)
        return (v[0], v[2], v[1] - (a // b) * v[2])


def modular_linear_solver(a:int, b:int, n:int) -> tuple:
    """
        This is algorithm for solving equation like this
        ax = b (mod n)
    Args:
        a (int): a in equation
        b (int): b in equation
        n (int): n in equation

    Complexity: O(lg(n) + gcd(a, n))

    Returns:
        tuple: a tuple of solutions. if equation have not solutions, will be returned empty tuple
    """
    v = euclid_extended(a, n)
    if b % v[0] == 0:
        x = (v[1] *(b//v[0])) % n
        lst = [x + i*(n//v[0]) for i in range(v[0])]
        return tuple(lst)
    else:
        return tuple()


def modular_exponentiation(a:int, b:int, n:int) -> int:
    """
    Imagine that you must solve this: a^b = x ( mod n)
    This algorithm find x for you.

    Complexity: O( len(go_to_binary(b)) ) + O(bn) where bn - count of digits in b

    Args:
        a (int): a in equation
        b (int): b in equation
        n (int): n in equation

    Returns:
        int: x in equation
    """
    c = 0
    d = 1
    bi = go_to_binary(b)
    k = len(bi)
    for i in range(k - 1,-1,-1):
        c = c * 2
        d = (d * d) % n
        if bi[i] == "1":
            c = c + 1
            d = (d * a) % n
    return d


# Sorting algorithms
def bubleSort(lst:list, reverse:bool=False, copy:bool=True) -> list:
    """
        Buble sort. Naive realisation.
        Ascending sort by default.
        Set reverse=True, if you want descending order

    Complexity: O(n^2)

    Default using:
        a = [2,1,5,3,9,10,3,5,4]
        a = bubleSort(a)
        If you want use like this:
        a = [2,1,5,3,9,10,3,5,4]
        bubleSort(a)
        Set copy=False and use it like this:
        a = [2,1,5,3,9,10,3,5,4]
        bubleSort(a, copy=False)

    Args:
        lst (list): unsorted list
        reverse (bool, optional): ascending/descending order. Defaults to False.
        copy (bool, optional): copy/nocopy. Defaults to True.

    Returns:
        list: sorted list
    """
    def wrongOrder(a,b):
        return a < b if reverse else a > b

    a = lst.copy() if copy else lst
    for j in range(len(a)-1):
        for i in range(len(a)-1):
            if wrongOrder(a[i], a[i+1]):
                c = a[i+1]
                a[i+1] = a[i]
                a[i] = c
    return a


def bubleSort_optimized(lst:list, reverse:bool=False, copy:bool=True) -> list:
    """
        Buble sort. Optimized realisation.
        Ascending sort by default.
        Set reverse=True, if you want descending order

    Complexity: O(n^2)

    Default using:
        a = [2,1,5,3,9,10,3,5,4]
        a = bubleSort(a)
        If you want use like this:
        a = [2,1,5,3,9,10,3,5,4]
        bubleSort(a)
        Set copy=False and use it like this:
        a = [2,1,5,3,9,10,3,5,4]
        bubleSort(a, copy=False)

    Args:
        lst (list): unsorted list
        reverse (bool, optional): ascending/descending order. Defaults to False.
        copy (bool, optional): copy/nocopy. Defaults to True.

    Returns:
        list: sorted list
    """
    def wrongOrder(a,b):
        return a < b if reverse else a > b

    a = lst.copy() if copy else lst
    j = 0
    flag = True
    while flag:
        flag = False
        for i in range(len(a)-j-1):
            if wrongOrder(a[i], a[i+1]):
                c = a[i+1]
                a[i+1] = a[i]
                a[i] = c
                flag = True
        j += 1
    return a


# string algorithms
def prefix_function_naive(s:str) -> list:
    """
        This is naive algorithm for computing prefix function.
        prefix function is a collection p where p[i] = max{k | s[0...k-1] == s[i-k+1...i]}
                                                   k=1..i

    Complexity: O(n^3) where n = len(s)

    Args:
        s (str): input str

    Returns:
        list: prefix function
    """
    n = len(s)
    p = [0] * n
    for i in range(n):
        m = 0
        for k in range(1, i + 1):
            j = 0
            while j < k and s[j] == s[j + i - k + 1]:
                j += 1
            if j == k and m < k:
                m = k
        p[i] = m
    return p


def prefix_function_optimized(s:str) -> list:
    """
        This is naive algorithm for computing prefix function.
        prefix function is a collection p where p[i] = max{k | s[0...k-1] == s[i-k+1...i]}
                                                   k=1..i
        Optimization of naive algorithm consists of two properties:
        1. for each i in [0..n-2]: p[i] + 1 >= p[i + 1]
        2. if s[i + 1] == s[p[i]] then p[i + 1] = p[i] + 1
        With this optimizations algorithm have complexity O(n) but naive algorithm have O(n^3) !

    Complexity: O(n) where n = len(s)

    Args:
        s (str): input str

    Returns:
        list: prefix function
    """
    n = len(s)
    p = [0] * n
    for i in range(1, n):
        j = p[i - 1]
        while j > 0 and s[i] != s[j]:
            j = p[j - 1]
        j = j + 1 if s[i] == s[j] else j
        p[i] = j
    return p


def z_function_naive(s:str) -> list:
    """
        This is algorithm for computing z function.
        Algorithm returns collection z.
        z[i] - наибольший общий префикс строки s и её i-го суффикса
        This is naive realisation of this algorithm.

    Complexity: O(n^2)

    Args:
        s (str): input string

    Returns:
        list: collection z
    """
    n = len(s)
    z = [0] * n
    for i in range(1, n):
        while s[i + z[i]] == s[z[i]] and i + z[i] < n:
            z[i] += 1
    return z


def z_function_optimized(s:str) -> list:
    """
        This is algorithm for computing z function.
        Algorithm returns collection z.
        z[i] - наибольший общий префикс строки s и её i-го суффикса
        This is optimized realisation of this algorithm.

    Complexity: O(n)

    Args:
        s (str): input string

    Returns:
        list: collection z
    """
    n = len(s)
    z = [0] * n
    l = 0
    r = 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if (i + z[i] - 1 > r):
            l = i
            r = i + z[i] - 1
    return z


def strCompession_Zfunc(s:str) -> str:
    """
        This algorithm find string such string t that s = t + t + ... + t.
        This algorithm uses Z-function optimized realisation.

    Complexity: O(n)

    Args:
        s (str): input string

    Returns:
        str: compressed string
    """
    n = len(s)
    z = z_function_optimized(s)
    for i in range(n):
        if i + z[i] == n and n % i == 0:
            t = s[:i]
            return t
    return None


def strCountDifferentSubstr_Zfunc(s:str) -> int:
    """
        This algorithm find for you count of different substrings in string s by Z-function.

    Complexity: O(n^2)

    Args:
        s (str): input string

    Returns:
        int: count of different substrings
    """
    def z_func(s:str):
        n = len(s)
        z = [0] * n
        l = 0
        r = 0
        z_max = 0
        for i in range(1, n):
            if i <= r:
                z[i] = min(r - i + 1, z[i - l])
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1
            if (i + z[i] - 1 > r):
                l = i
                r = i + z[i] - 1
            if z_max < z[i]:
                z_max = z[i]
        return z_max

    n = len(s)
    string = ""
    count = 0
    for i in range(n):
        string += s[i]
        length = i + 1
        z = z_func(string)
        count += length - z
    return count


def strFind_naive(pattern:str, text:str) -> int:
    """
       This is naive realisation algorithm
       for finding a substring in a string

    Complexity: O((n - m + 1)m) where n = len(text), m = len(pattern)

    Args:
        pattern (str): pattern
        text (str): text for finding matches

    Returns:
        int: count of matches
    """
    s = 0
    m = len(pattern)
    n = len(text)
    for i in range(n-m+1):
        j = 0
        flag = True
        while flag and j < m:
            if text[i+j] != pattern[j]:
                flag = False
            j += 1
        if flag:
            s += 1
    return s


def strFind_RabinKarp(pattern:str, text:str, q:int = 101, d:int = 256, table:dict = None) -> int:
    """This is Rabin-Karp algorithm for
       finding a substring in a string

    Complexity:
                Preprocessing: θ(m)
                Comparisons:   O((n - m + 1)m)
                Total:         θ(m) + O((n - m + 1)m)
    Args:
        pattern (str): pattern
        text (str): text for finding matches
        q (int): prime number. dq should fit into a computer word
        d (int): alphabet power
        table (dict): comparison table. It's must look like {'a': 0, 'b': 1, 'c': 2, ...}
    Returns:
        int: count of matches
    """
    m = len(pattern)
    n = len(text)
    h = 1
    p = 0
    t = 0
    count = 0

    # h = d^(m-1) mod q
    for i in range(m-1):
        h = (h * d) % q

    if d == 256 and table is None:
        # creating p and t_0
        for i in range(m):
            p = (ord(pattern[i]) + d * p) % q
            t = (ord(text[i]) + d * t) % q

        for s in range(n - m + 1):
            if p == t:
                j = 0
                flag = True
                while flag and j < m:
                    if text[s+j] != pattern[j]:
                        flag = False
                    j += 1
                if flag:
                    count += 1
            if s < n - m:
                # getting t_s+1 throught t_s
                t = (d * (t - h * ord(text[s])) + ord(text[s + m])) % q
                if t < 0:
                    t = t + q
        return count
    else:
        # creating p and t_0
        for i in range(m):
            p = (table(pattern[i]) + d * p) % q
            t = (table(text[i]) + d * t) % q

        for s in range(n - m + 1):
            if p == t:
                j = 0
                flag = True
                while flag and j < m:
                    if text[s+j] != pattern[j]:
                        flag = False
                    j += 1
                if flag:
                    count += 1
            if s < n - m:
                # getting t_s+1 throught t_s
                t = (d * (t - h * table(text[s])) + table(text[s + m])) % q
                if t < 0:
                    t = t + q
        return count


def strFind_Zfunc(pattern:str, text:str) -> int:
    """
       This is realisation of algorithm
       for finding a substring in a string
       with using Zfunction

    Complexity: O(len(pattern) + len(text))

    Args:
        pattern (str): pattern
        text (str): text for finding matches

    Returns:
        int: count of matches
    """
    s = pattern + "#" + text
    z = z_function_optimized(s)
    p = len(pattern)
    count = 0
    for i in range(len(text)):
        if z[i + p + 1] == p:
            count += 1
    return count


def strFind_KMP(pattern:str, text:str) -> int:
    """
       This is realisation of Knuth-Morris-Pratt algorithm
       for finding a substring in a string.
       This algorithm uses prefix_function_optimized
       function from this library.

    Time complexity: O(n + m) where n = len(pattern), m = len(text)
    Memory complexity: O(n + m)

    Args:
        pattern (str): pattern
        text (str): text for finding matches

    Returns:
        int: count of matches
    """
    n = len(pattern)
    w = pattern + '#' + text
    p = prefix_function_optimized(w)
    count = 0
    for i in range(n + 1, len(p)):
        if p[i] == n:
            count += 1
    return count


def strFind_KMP_bonus(pattern:str, text:str) -> int:
    """
       This is realisation of Knuth-Morris-Pratt algorithm
       for finding a substring in a string.
       This algorithm uses prefix_function_optimized
       function from this library.
       This realisation have memory complexity: O(n)

    Time complexity: O(n + m) where n = len(pattern), m = len(text)
    Memory complexity: O(n)

    Args:
        pattern (str): pattern
        text (str): text for finding matches

    Returns:
        int: count of matches
    """
    n = len(pattern)
    m = len(text)
    P = prefix_function_optimized(pattern)
    # entries = [] // uncommented commented strings for finding entries
    i, j, count = 0, 0, 0
    while i < m:
        if text[i] == pattern[j]:
            if j == n - 1:
                # entries.append(i - n + 1)
                count += 1
                j = P[j]
            else:
                j += 1
            i += 1
        elif j:
            j = P[j-1]
        else:
            i += 1
    return count


def strHash(s:str, p:int = 31) -> int:
    """
        This is algorithm for hashing strings.
        h(s) = s[0] + s[1] * p + s[2] * p^2 + ... + s[n] * p^n
        Where s[n] - code of char s[n] in Unicode, p - prime number

    Complexity: O(n) where n = len(s)

    Args:
        s (str): input string

    Returns:
        int: hash of string s
    """
    a = [ord(s[i]) - ord('a') + 1 for i in range(len(s))]
    return horners_rule(a, p)


#graphs
def get_graph_listadjacency(lst:list, numbers:bool=True) -> list | dict:
    """
        This algorithm for create a list of adjancency by list of edges.
        Edges in list of edges a must be a tuples of this form (a, b).
        (a, b) means that the edge starts in vertex a and ends in vertex b.
        By default I suggest that a and b is int such that a,b >= 0.
        If a and b is not int set numbers=False and you get a dict of adjancency.
        If you set numbers=False then I suggest that a,b is not mutable
        and using this function you will get an adjacency dictionary.
        I suggest that you are using this algorithm for undirected unweighted graph but
        this also works for directed unweighted graph

    Complexity: O(m) where m - count of edges in graph

    Args:
        a (list): list of edges. Edges must be a tuples of this form (a, b).
        numbers (bool, optional): set False if a or b is not int such that a,b >= 0. Defaults to True.

    Returns:
        list | dict: list of adjancency | dict of adjancency
    """
    if numbers:
        n = 0
        for (a, b) in lst:
            mx = max(a, b)
            if mx > n:
                n = mx
        adjlst = [[] for i in range(n + 1)]
        for (a, b) in lst:
            adjlst[a].append(b)
        return adjlst
    else:
        d = {}
        for (a, b) in lst:
            if not (a in d.keys()):
                d[a] = []
            d[a].append(b)
        return d


def get_graph_matrixadjacency(lst:list, numbers=True) -> list | dict:
    """
        This algorithm for create a matrix of adjancency by list of edges.
        Edges in list of edges a must be a tuples of this form (a, b).
        (a, b) means that the edge starts in vertex a and ends in vertex b.
        By default I suggest that a and b is int such that a,b >= 0.
        If a and b is not int set numbers=False and you get a dict of adjancency.
        If you set numbers=False then I suggest that a,b is not mutable
        and using this function you will get an adjacency dictionary.
        I suggest that you are using this algorithm for undirected unweighted graph but
        this also works for directed unweighted graph

    Complexity: O(m^2) where m - count of edges

    Args:
        lst (list): list of edges. Edges must be a tuples of this form (a, b).
        numbers (bool, optional): set False if a or b is not int such that a,b >= 0. Defaults to True.

    Returns:
        list | dict: matrix of adjancency | matrix-dict of adjancency
    """
    if numbers:
        n = 0
        for (a, b) in lst:
            mx = max(a, b)
            if mx > n:
                n = mx
        matrix = [[0 for j in range(n + 1)] for i in range(n + 1)]
        for (a, b) in lst:
            matrix[a][b] = 1
        return matrix
    else:
        d = {a: {a: 0 for (a, b) in lst} for (a, b) in lst}
        for (a, b) in lst:
            d[a][b] = 1
        return d


def get_weightgraph_listadjacency(lst:list, numbers:bool=True) -> list | dict:
    """
        This algorithm for create a list of adjancency by list of edges.
        Edges in list of edges a must be a tuples of this form (a, b, w).
        (a, b, w) means that the edge starts in vertex a and ends in vertex b and edge have weight w.
        By default I suggest that a and b is int such that a,b >= 0.
        If a and b is not int set numbers=False and you get a dict of adjancency.
        If you set numbers=False then I suggest that a,b is not mutable
        and using this function you will get an adjacency dictionary.
        I suggest that you are using this algorithm for undirected unweighted graph but
        this also works for directed weighted graph

    Complexity: O(m) where m is count of edges

    Args:
        lst (list): list of edges. Edges must be a tuples of this form (a, b, w). w - weight
        numbers (bool, optional): set False if a or b is not int such that a,b >= 0. Defaults to True.

    Returns:
        list | dict: list of adjancency | dict of adjancency
    """
    if numbers:
        n = 0
        for (a, b, w) in lst:
            mx = max(a, b)
            if mx > n:
                n = mx
        adjlst = [[] for i in range(n + 1)]
        for (a, b, w) in lst:
            adjlst[a].append((b, w))
        return adjlst
    else:
        d = {}
        for (a, b, w) in lst:
            if not (a in d.keys()):
                d[a] = []
            d[a].append((b, w))
        return d


def get_weightgraph_matrixadjacency(lst:list, numbers=True) -> list | dict:
    """
        This algorithm for create a matrix of adjancency by list of edges.
        Edges in list of edges a must be a tuples of this form (a, b, w).
        (a, b, w) means that the edge starts in vertex a and ends in vertex b and edge have weight w..
        By default I suggest that a and b is int such that a,b >= 0.
        If a and b is not int set numbers=False and you get a dict of adjancency.
        If you set numbers=False then I suggest that a,b is not mutable
        and using this function you will get an adjacency dictionary.
        I suggest that you are using this algorithm for undirected unweighted graph but
        this also works for directed weighted graph

    Complexity: O(m^2) where m - count of edges

    Args:
        lst (list): list of edges. Edges must be a tuples of this form (a, b, w).
        numbers (bool, optional): set False if a or b is not int such that a,b >= 0. Defaults to True.

    Returns:
        list | dict: matrix of adjancency | matrix-dict of adjancency
    """
    if numbers:
        n = 0
        for (a, b, w) in lst:
            mx = max(a, b)
            if mx > n:
                n = mx
        matrix = [[-1 for j in range(n + 1)] for i in range(n + 1)]
        for (a, b, w) in lst:
            matrix[a][b] = w
        return matrix
    else:
        d = {a: {a: -1 for (a, b) in lst} for (a, b) in lst}
        for (a, b, w) in lst:
            d[a][b] = w
        return d


def get_empty_color(lst:list | dict) -> list | dict:
    """
        This is helper function for dfs algorithms.
        With her help dfs algorithms gets an empty list | dict of vertex colors.

    Complexity: O(n) where n - count of vertex in graph

    Args:
        lst (list | dict): list/dict of adjacency

    Raises:
        TypeError: Wrong arguments

    Returns:
        list | dict: list/dict of vertex colors
    """

    if isinstance(lst, list):
        n = len(lst)
        color = [0 for i in range(n)]
        return color

    if isinstance(lst, dict):
        color = {}
        for obj in lst.keys():
            color[obj] = 0
        return color

    raise TypeError("Wrong arguments")


def graph_dfs(lst:list | dict, color:list | dict, u:Any, weight:bool = False):
    """
        This is algorithm of depth-walk search which start from vertex of undirected/directed unweight graph.
        I suggest that you are using for this algorithm a list/dict of adjacency which you
        can create by a function get_graph_listadjacency.
        color[i] must be one of this int - 0, 1, 2.
        color must be the same type as lst.
        u must be not mutable

    Complexity: O(m) where m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        color (list | dict): color of vertex
        u (Any): vertex from which dfs is statring
        weight (bool): set weight = True if your graph are weighted. Defaults to False.
    """
    if weight:
        color[u] = 1
        for (v, w) in lst[u]:
            if color[v] == 0:
                graph_dfs(lst, color, v, weight)
        color[u] = 2
    else:
        color[u] = 1
        for v in lst[u]:
            if color[v] == 0:
                graph_dfs(lst, color, v)
        color[u] = 2


def graph_dodfs(lst:list | dict, weight:bool = False) -> list | dict:
    """
        This is algorithm of depth-walk search on undirected/directed unweight graph.
        I suggest that you are using for this algorithm a list/dict of adjacency which you
        can create by a function get_graph_listadjacency.

    Complexity: O(n + m)
                where n - count of vertex in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        weight (bool): set weight = True if your graph are weighted. Defaults to False.

    Raises:
        TypeError: Wrong arguments

    Returns:
        list | dict: list/dict of colors - the result of walking
    """
    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    color = get_empty_color(lst)
    array = range(len(lst)) if condition_list else lst.keys()

    for u in array:
        graph_dfs(lst, color, u, weight)

    return color


def get_all_connected_components(lst: list| dict, weight:bool = False) -> tuple:
    """
        This algorithm find all connected components in your weighted/unweighted graph.
        This algorithm returns a tuple of this form: (component, count)
        where component: list | dict such that component[v] = number connected component.
              count - count connected components in your graph
        This algorithm based on dfs

    Complexity: O(n + m)
                where n - count of vertexes in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        weight (bool, optional): set True if you have weighted graph. Defaults to False.

    Raises:
        TypeError: Wrong arguments

    Returns:
        tuple: (component, count)
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    def advanced_dfs(v:Any, num:int, lst: list | dict, component:list | dict, weight:bool = False):
        component[v] = num
        if weight:
            for (u, w) in lst[v]:
                if component[u] == 0:
                    advanced_dfs(u, num, lst, component, weight)
        else:
            for u in lst[v]:
                if component[u] == 0:
                    advanced_dfs(u, num, lst, component, weight)


    array = range(len(lst)) if condition_list else lst.keys()
    component = [0 for i in range(len(lst))] if condition_list else {obj: 0 for obj in lst.keys()}

    num = 0

    for v in array:
        if component[v] == 0:
            num += 1
            advanced_dfs(v, num, lst, component, weight)

    return (component, num)


def is_bipartite(lst: list|dict, weight:bool = False) -> tuple | bool:
    """
        This is algorithm for determine bipartite of graph.
        If your graph is bipartite then you get tuple of this form: (True, color)
        True - bool that tell you graph is bipartite
        color - list | dict - this is coloring book for each vertex in your graph
        This algorithm based on dfs

    Complexity: O(n + m)
                where n - count of vertexes in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        weight (bool, optional): Set True if your graph are weighted. Defaults to False.

    Raises:
        TypeError: Wrong arguments

    Returns:
        tuple | bool: (True, color) | False
    """
    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    def advanced_dfs(v:Any, col:int, lst: list | dict, color:list | dict, weight:bool = False):
        color[v] = col
        if weight:
            for (u, w) in lst[v]:
                if color[u] == 0:
                    result = advanced_dfs(u, -col, lst, color, weight)
                    if isinstance(result, bool):
                        return False
                elif color[u] != - col:
                    return False
        else:
            for u in lst[v]:
                if color[u] == 0:
                    result = advanced_dfs(u, -col, lst, color, weight)
                    if isinstance(result, bool):
                        return False
                elif color[u] != - col:
                    return False

    array = range(len(lst)) if condition_list else lst.keys()
    color = [0 for i in range(len(lst))] if condition_list else {obj: 0 for obj in lst.keys()}

    for v in array:
        if color[v] == 0:
            result = advanced_dfs(v, 1, lst, color, weight)
            if isinstance(result, bool):
                return False

    return (True, color)


def have_cycle(lst: list|dict, weight:bool = False) -> tuple:
    """
        This algorithm find one cycle in graph.
        Use this algorithm if you need to determine a tree
        This algorithm based of dfs

    Complexity: O(n + m)
                where n - count of vertexes in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        weight (bool, optional): Set True if your graph are weighted. Defaults to False.

    Raises:
        TypeError: Wrong Arguments

    Returns:
        tuple: (bool, cycle)
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    def alternative_dfs(v:Any,used:list | dict, lst: list | dict, p:Any = -1, weight:bool=False, flg:lst=[]) -> list:
        if used[v]:
            s = 'cycle: '
            s += str(v)
            return [True, s, v]

        used[v] = 1

        if weight:
            for (u, w) in lst[v]:
                try:
                    if u != p:
                        k = alternative_dfs(u, used, lst, p=v, weight=weight, flg=flg)
                        if len(flg) == 0 and k[2] != -1:
                            k[1] += ' <- ' + str(v)
                            if k[2] == v:
                                # exit(0)
                                flg.append(1)
                                raise CycleFound
                            return k
                        else:
                            return k
                except CycleFound:
                    return k
        else:
            for u in lst[v]:
                try:
                    if u != p:
                        k = alternative_dfs(u,used, lst, p=v ,weight=weight, flg=flg)
                        if len(flg) == 0 and k[2] != -1:
                            k[1] += ' <- ' + str(v)
                            if k[2] == v:
                                # exit(0)
                                flg.append(1)
                                raise CycleFound
                            return k
                        else:
                            return k
                except CycleFound:
                    return k

        return [False, '', -1]


    array = range(len(lst)) if condition_list else lst.keys()
    used = [0 for i in range(len(lst))] if condition_list else {obj: 0 for obj in lst.keys()}

    for v in array:
        if used[v] == 0:
            k = alternative_dfs(v, used, lst, weight=weight)
            if k[0]:
                return (k[0], k[1])
    return (False, '')


def topological_sort(lst: list | dict, weight:bool = False) -> list:
    """
        This is algorithm of topological sorting of graph
        Whis algorithm works for acyclic graph
        Algorithm returns list t - topological order of vertexes in graph.
        topological orders means:
        Let t[i] = u, t[j] = v.
        for each i,j: i < j we have no path from v to u

        This algorithm baased on dfs

    Complexity: O(n + m)
                where n - count of vertexes in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        weight (bool, optional): Set True if your graph are weighted. Defaults to False.

    Raises:
        TypeError: Wrong arguments

    Returns:
        list: topological order
    """
    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    def dfs(v:Any, lst: list | dict, used: list | dict, t:list):
        used[v] = 1
        if weight:
            for (u, w) in lst[v]:
                if used[u] == 0:
                    dfs(u, lst, used, t)
            t.append(v)
        else:
            for u in lst[v]:
                if used[u] == 0:
                    dfs(u, lst, used, t)
            t.append(v)

    array = range(len(lst)) if condition_list else lst.keys()
    used = [0 for i in range(len(lst))] if condition_list else {obj: 0 for obj in lst.keys()}
    t = []

    for v in array:
        if used[v] == 0:
            dfs(v, lst, used, t)
    t.reverse()
    return t


def graph_bfs(lst:list | dict, s:Any, weight:bool = False) -> tuple:
    """
        This algorithm of bfs on graph.
        Input graph must be unweighted
        This algorithm returns tuple of this form (visited, d, p)
        visited - array of visited vertexes
        d - array of shortest paths from vertex s. d[v] - shortest path from s to v
        p - array of parrents. p[v] - parrent of vertex v

    Complexity: O(n + m)
                where n - count of vertex
                      m - count of edges

    Args:
        lst (list | dict): list/dict of adjacency
        s (Any): Start vertex
        weight (bool, optional): set True, if you want to get expirement. Defaults to False.

    Returns:
        tuple: (visited, d, p)
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrond arguments')

    q = Queue()
    q.put(s)

    if condition_list:
        n = len(lst)
        visited = [False for i in range(n)]
        p = [None for i in range(n)]
        d = [-1 for i in range(n)]
    else:
        visited, p, d = {}, {}, {}
        for obj in lst.keys():
            visited[obj] = False
            p[obj] = None
            d[obj] = -1

    visited[s] = True
    d[s] = 0

    while not (q.empty()):
        v = q.get()
        if weight:
            for (u, w) in lst[v]:
                if not visited[u]:
                    q.put(u)
                    visited[u] = True
                    d[u] = d[v] + w
                    p[u] = v
        else:
            for u in lst[v]:
                if not visited[u]:
                    q.put(u)
                    visited[u] = True
                    d[u] = d[v] + 1
                    p[u] = v

    return (visited, d, p)


def dejcstra(lst:list | dict, s:Any, max_weight:Any = None) -> list | dict:
    """
        This is Dejcsta's algorithm for findind shortest paths from s to other vertexes in weighted graph.
        Attention: weight of each edge must be >= 0.
        If you know the maximal weight of edge in graph (let it be max), set max_weight = max + 1

        Algotihm:
            Start:
                d[s] = 0 , a[v] = 0
                for each v in V\{s}: d[v] = infinity
             Basis:
                1. Choose v:
                    v = min{d[u]: a[u] == 0}
                2. a[v] = 1
                3. decompression:
                    1. for each (v, u, w):
                        d[u] = min(d[u], d[v] + w)

    Complexity: O(n^2) where n - count vertexes in graph

    Args:
        lst (list | dict): list/dict of adjacency
        s (Any): start vertex. s is not mutable
        max_weight (Any, optional): max weight + 1. Defaults to None.

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list | dict: array of minimal weight paths from vertex s to other vertexes in graph
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    max_weight = max_weight if not ( max_weight is None ) else 1.8446744e+19
    n = len(lst)

    a = [False for i in range(n)] if condition_list else {obj: 0 for obj in lst.keys()}
    d = [max_weight for i in range(n)] if condition_list else {obj: max_weight for obj in lst.keys()}

    # if condition_list:
    #     a = [False for i in range(n)]
    #     d = [max_weight for i in range(n)]
    # else:
    #     a, d = {}, {}
    #     for i in lst.keys():
    #         a[i] = 0
    #         d[i] = max_weight

    d[s] = 0

    array = range(n) if condition_list else lst.keys()
    for i in range(n):
        v = -1 if condition_list else None
        for u in array:
            if (not a[u]) and ( (v in (-1, None)) or d[u] < d[v]):
                v = u

        a[v] = True

        for (u, w) in lst[v]:
            d[u] = min(d[u], d[v] + w)

    return d


def dejcstra_heap(lst:list | dict, s:Any, max_weight:Any = None) -> list | dict:
    """
        This is Dejcsta's algorithm for findind shortest paths from s to other vertexes in weighted graph.
        Attention: weight of each edge must be >= 0.
        If you know the maximal weight of edge in graph (let it be max), set max_weight = max + 1

        This is improved version of this algorithm.
        For finding v = min{d[u]: a[u] == 0} is using priority queue which based on heap

    Complexity: O(nlogn + m)
                where n - count of vertex in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): _description_
        s (Any): _description_
        max_weight (Any, optional): _description_. Defaults to None.

    Raises:
        TypeError: _description_

    Returns:
        list | dict: _description_
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    max_weight = max_weight if not ( max_weight is None ) else 1.8446744e+19
    n = len(lst)

    if condition_list:
        d = [max_weight for i in range(n)]

    else:
        d = {}
        for i in lst.keys():
            d[i] = max_weight

    d[s] = 0

    q = PriorityQueue()
    q.put((0, s))
    while not q.empty():
        (cur_d, v) = q.get()
        if cur_d > d[v]:
            continue
        for (u, w) in lst[v]:
            if d[u] > d[v] + w:
                d[u] = d[v] + w
                q.put((d[u], u))

    return d


# data structures
class CycleFound(Exception):

    def __init__(self):
        super().__init__()


class Heap:
    """
        This is a heap.
        Heap is using in heap-sort algorithm.
        You can create heap: h = Heap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        Look at the result: h.A
        You can create min-heap: h = Heap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7], max_heap = False)
    """

    def __init__(self, a:list, max_heap = True):
        self.A = self.__build_max_heap(a) if max_heap else self.__build_min_heap(a)


    def __max_heapify(self, a:list, i:int):
        """
            Complexity: O(log(n))
        """

        L = 2*i + 1
        R = 2*i + 2
        largest = L if L < len(a) and a[L] > a[i] else i
        largest = R if R < len(a) and a[R] > a[largest] else largest
        if largest != i:
            swap = a[largest]
            a[largest] = a[i]
            a[i] = swap
            self.__max_heapify(a, largest)


    def __build_max_heap(self, a:list):
        """
            Complexity: O(n)
        """

        heap = a.copy()
        for i in range((len(a) // 2) + 1, -1, -1):
            self.__max_heapify(heap, i)
        return heap


    def __min_heapify(self, a:list, i:int):
        """
            Complexity: O(log(n))
        """

        L = 2*i + 1
        R = 2*i + 2
        minest = L if L < len(a) and a[L] < a[i] else i
        minest = R if R < len(a) and a[R] < a[minest] else minest
        if minest != i:
            swap = a[minest]
            a[minest] = a[i]
            a[i] = swap
            self.__min_heapify(a, minest)


    def __build_min_heap(self, a:list):
        """
            Complexity: O(n)
        """

        heap = a.copy()
        for i in range((len(a) // 2) + 1, -1, -1):
            self.__min_heapify(heap, i)
        return heap


class Hash_table:
    """
        This is hash_table with open addressing
        Collisions are solved by double hashing
        The main hash function uses universal hashing
    """

    def __init__(self, *args):
        self.n = len(args)
        self.count = self.n
        self.__P = 701
        self.__A = randint(1, self.__P - 1)
        self.__B = randint(0, self.__P - 1)
        if self.n == 0:
            self.m = 10
            self.table = [None for i in range(self.m)]
            self.alpha = 0
        else:
            self.m = 2 * self.n
            self.table = [None for i in range(self.m)]
            self.alpha = self.n / self.m
            for element in args:
                key = element[0]
                self.insert(element)


    def __hash_func(self, key:int | float | bool | str, i:int):
        if isinstance(key, (float, bool)):
            key = int(key)

        if isinstance(key, str):
            key = strHash(key)

        def h(key):
            return ((self.__A * key + self.__B) % self.__P ) % self.m

        def g(key):
            return 1 + key % (self.m - 1)

        return (h(key) + i * g(key)) % self.m


    def insert(self, x:tuple):

        def default(x, flg = True):
            key = x[0]
            i = 0
            while not (self.table[self.__hash_func(key, i)] is None):
                i += 1
            self.table[self.__hash_func(key, i)] = x
            if flg:
                self.n += 1
                self.count = self.n
                self.alpha = self.n / self.m

        if self.alpha < 0.7:
            default(x)
        else:
            args = []
            for i in range(self.m):
                if not (self.table[i] is None):
                    args.append(self.table[i])
            self.m = int((10 * (self.n + 1)) / 3)
            self.table = [None] * self.m
            for element in args:
                default(element, flg = False)
            default(x)


    def _find(self, key:int | float | bool | str):
        i = 0
        x = ((), ())
        while key != x[0]:
            ikey = self.__hash_func(key, i)
            x = self.table[ikey]
            if x is None:
                raise KeyError('Inccorect key has been put')
            i += 1
        return (ikey, x[1])


    def __setitem__(self, key:int | float | bool | str, value):
        try:
            ikey = self._find(key)[0]
            self.table[ikey] = (key, value)
        except KeyError:
            self.insert((key, value))


    def __getitem__(self, key:int | float | bool | str):
        value = self._find(key)[1]
        return value


    def delete(self, key:int | float | bool | str):
        const = "DELETED"
        ikey = self._find(key)
        self.table[ikey] = const
        self.count -= 1


    def __str__(self):
        def get_string(s:int | float | bool | str):
            if isinstance(s, str):
                return '\'' + s + '\''
            if isinstance(s, int | float | bool):
                return str(s)

        string = "{ "
        c = 0
        for x in self.table:
            if not (x is None) and not (x == "DELETED"):
                if c == self.count - 1:
                    string += get_string(x[0]) + " : " + get_string(x[1]) + " "
                else:
                    string += get_string(x[0]) + " : " + get_string(x[1]) + ", "
                c += 1
        string += "}"
        return string

    __repr__ = __str__


class RedBlackTree:
    pass


class Poly:

    def __init__(self, *args):

        if len(args) == 1 and isinstance(args[0], (int, float)):
            self.lst = [args[0]]

        if len(args) == 1 and (not  isinstance(args[0], (int, float)) ) :
            self.lst = [element for element in args[0]]

        if len(args) > 1:
            self.lst = [element for element in args]

        if len(args) == 0:
            self.lst =[]

    def __repr__(self):
        length = len(self.lst)
        if length == 0:
            return '\'Poly((0))\''
        if length == 1:
            return f'\'Poly(({self.lst[0]}))\''
        return '\'Poly(' + str(tuple(self.lst)) + ')\''

    def __str__(self):
        strinG = ''
        length = len(self.lst)
        if length != 0:
            n = 0
            for i in self.lst:
                if i == 0:
                    n += 1
            if n == length:
                return '0'
            if length == 1:
                value = self.lst[0]
                cost = f'{abs(round(value, 3))}'
                if value < 0:
                    cost = '-' + cost
                strinG += cost
                return strinG
            for i in range(length - 1, -1, -1):
                value = self.lst[i]
                if value != 0:

                    if strinG == '':

                        cost = ('x' if i == 1 else f'x^{i}') if abs(value) == 1 else (f'{abs(round(value, 3))}x' if i == 1 else f'{abs(round(value, 3))}x^{i}')

                        if value < 0:
                            cost = '-' + cost

                    else:

                        cost = (('1' if i == 0 else 'x') if (i == 0 or i == 1) else f'x^{i}') if (abs(value) == 1) else ((f'{abs(round(value, 3))}' if i == 0 else f'{abs(round(value, 3))}x') if (i == 0 or i == 1) else (f'{abs(round(value, 3))}x^{i}'))

                        if value < 0:
                            cost = ' - ' + cost
                        if value > 0:
                            cost = ' + ' + cost

                    strinG += cost

            return strinG
        else:
            return '0'


    def __add__(self, other):

        if isinstance(other, (int, float)):
            if len(self.lst) != 0:
                lst = self.lst.copy()
                lst[0] += other
                return self.__class__(lst)
            else:
                return self.__class__(other)

        if isinstance(other, Poly):

            if len(self.lst) != 0 and len(other.lst) != 0:
                max_len = max(len(self.lst), len(other.lst))
                lst = [0] * max_len
                for i in range(len(self.lst)):
                    lst[i] += self.lst[i]
                for i in range(len(other.lst)):
                    lst[i] += other.lst[i]
                return self.__class__(lst)

            if len(self.lst) == 0 and len(other.lst) != 0:
                return self.__class__(other.lst)

            if len(self.lst) != 0 and len(other.lst) == 0:
                return self.__class__(self.lst)

            if len(self.lst) == 0 and len(other.lst) == 0:
                return self.__class__()

        raise TypeError('can\'t add')


    def __sub__(self, other):

        if isinstance(other, (int, float)):
            if len(self.lst) != 0:
                lst = self.lst.copy()
                lst[0] -= other
                return self.__class__(lst)
            else:
                return self.__class__(-other)

        if isinstance(other, Poly):
            if len(self.lst) != 0 and len(other.lst) != 0:
                max_len = max(len(self.lst), len(other.lst))
                lst = [0] * max_len
                for i in range(len(self.lst)):
                    lst[i] += self.lst[i]
                for i in range(len(other.lst)):
                    lst[i] -= other.lst[i]
                return self.__class__(lst)

            if len(self.lst) == 0 and len(other.lst) != 0:
                return self.__class__(list(map(lambda x: -x, other.lst)))

            if len(self.lst) != 0 and len(other.lst) == 0:
                return self.__class__(self.lst)

            if len(self.lst) == 0 and len(other.lst) == 0:
                return self.__class__()

        raise TypeError('can\'t sub')


    def __eq__(self, other):
        if isinstance(other, (int, float)):
            if len(self.lst) == 1:
                return self.lst[0] == other
            if len(self.lst) == 0:
                return 0 == other
            return False

        if isinstance(other, Poly):
            return self.__str__() == other.__str__()

        raise TypeError('wrong type')


    def __rsub__(self, other): #other - self self = x^2+x+1, other = 1; other - self = - (self - other)
        return self.__class__(list(map(lambda x: -x, self.__sub__(other).lst)))



    def degree(self):
        if len(self.lst) != 0:
            for i in range(len(self.lst) - 1, -1 , -1):
                if self.lst[i] != 0:
                    return i
        else:
            return 0


    @staticmethod
    def poly_from_str(string:str):
        new_lst = []
        for element in string.split():
            if '.' in element:
                new_lst.append(float(element))
            else:
                new_lst.append(int(element))
        return Poly(new_lst)


    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_lst = [i*other for i in self.lst]
            return self.__class__(new_lst)

        if isinstance(other, self.__class__):
            length_self = len(self.lst)
            length_other = len(other.lst)
            new_lst = [0] * (length_self + length_other - 1)
            for i in range(length_self):
                for j in range(length_other):
                    new_lst[i+j] += self.lst[i]*other.lst[j]
            return self.__class__(new_lst)
        raise TypeError('can\'t multiply')

    __radd__ = __add__
    __rmul__ = __mul__


class DegreeIsTooBigException(Exception):

    def __init__(self, string:str):
        self.qpoly = string
        super().__init__()


class QuadraticPolynomial(Poly):

    def solve(self):
        if self.degree() > 2 :
            raise DegreeIsTooBigException(str(self))
        if self.degree() == 2:
            c = self.lst[0]
            b = self.lst[1]
            a = self.lst[2]
            D = b**2 - 4*a*c
            if D > 0:
                sqrt_D = D**(1/2)
                x1 = (-b+sqrt_D)/(2*a)
                x2 = (-b-sqrt_D)/(2*a)
                if x1 == int(x1): # 2.12431234234 == 2 - False    2.000000 == 2 True
                    x1 = int(x1)
                if x2 == int(x2):
                    x2 = int(x2)
                return sorted([x1,x2])
            if D == 0:
                x = (-b)/(2*a)
                if x == int(x):
                    x = int(x)
                return [x]
            if D < 0:
                return []


#sorting
def heapsort(lst:list, reverse:bool=False) -> list:
    """
        This is algorithm of sorting by heap.
        Ascending sort by default.
        Set reverse=True, if you want descending order

    Complexity: O(nlog(n)) where n = len(lst)

    Args:
        lst (list): unsorted list
        reverse (bool, optional): ascending/descending order. Defaults to False.

    Returns:
        list: sorted list
    """

    def maxheapify(a:list, i:int, heap_size:int):
        L = 2*i + 1
        R = 2*i + 2
        largest = L if L < heap_size and a[L] > a[i] else i
        largest = R if R < heap_size and a[R] > a[largest] else largest
        if largest != i:
            swap = a[largest]
            a[largest] = a[i]
            a[i] = swap
            maxheapify(a, largest, heap_size)

    def minheapify(a:list, i:int, heap_size:int):
        L = 2*i + 1
        R = 2*i + 2
        minest = L if L < heap_size and a[L] < a[i] else i
        minest = R if R < heap_size and a[R] < a[minest] else minest
        if minest != i:
            swap = a[minest]
            a[minest] = a[i]
            a[i] = swap
            maxheapify(a, minest, heap_size)

    a = Heap(lst, max_heap=(not reverse)).A

    size = len(a)
    for i in range(len(a) - 1, 0, -1):
        swap = a[0]
        a[0] = a[i]
        a[i] = swap
        size -= 1
        if reverse:
            minheapify(a, 0, size)
        else:
            maxheapify(a, 0, size)
    return a


if __name__ == "__main__":
    s = """
        Welcome to the my library of algorithms !
        Here you can find implementations of most
        fundamental algorithms and data structures.
        For each algorithm, you can find a brief
        description of it, as well Time and Memory Complexity.
    """
    print(s)
