from .math import horners_rule

# Optional:
# Поиск подстроки в строке с помощью автоматов

# future:
# Префиксное дерево
# Алгоритм Ахо-Корасик
# Дерево палиндромов
# Суффиксный массив

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


__all__ = [
    'prefix_function_naive',
    'prefix_function_optimized',
    'z_function_naive',
    'z_function_optimized',
    'strCompession_Zfunc',
    'strCountDifferentSubstr_Zfunc',
    'strFind_naive',
    'strFind_RabinKarp',
    'strFind_Zfunc',
    'strFind_KMP',
    'strFind_KMP_bonus',
    'strHash',
]
