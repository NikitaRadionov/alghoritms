from typing import Any
from random import randint
from .strings import strHash

class Dsu:
    """
        This is Disjoin Set Union.
        This implementation uses the heuristic of path compression and pooling by rank

        Complexity of each operation: O(a(n)) a(n) - inverse Ackermann function
        For simplicity, we can assume that we have constant complexity
    """

    def __init__(self):
        self.__storage = {}
        self.__rank = {}


    def make_set(self, x:Any):
        self.__storage[x] = x
        self.__rank[x] = 0


    def find_set(self, x:Any):
        if x == self.__storage[x]:
            return x
        p = self.find_set(self.__storage[x])
        self.__storage[x] = p
        return p


    def have_element(self, x:Any):
        try:
            return self.find_set(x)
        except KeyError:
            return False


    def union_sets(self, a:Any, b:Any):
        a = self.find_set(a)
        b = self.find_set(b)
        if a != b:
            if self.__rank[a] < self.__rank[b]:
                c = a
                a = b
                b = c
            self.__storage[b] = a
            if self.__rank[a] == self.__rank[b]:
                self.__rank[a] += 1


    def get_storage(self):
        x = self.__storage.copy()
        return x


    def get_rank(self):
        x = self.__rank.copy()
        return x


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


class HashTable:
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


class BinNode:
    """
        This is node of binary tree
        Use it for class BinTree.
    """

    def __init__(self, key:Any=None, p=None, left=None, right=None):
        self.key = key
        self.p = p
        self.left = left
        self.right = right


    def __str__(self):
        return str((self.key, self.left if self.left is None else self.left.key, self.right if self.right is None else self.right.key))


    __repr__ = __str__


class BinTree:
    """
        This is implementation of binary tree.

        Complexity:
                    insert: O(n)
                    delete: O(n)
                    search: O(log(n))
    """

    def __init__(self):
        self.storage = []
        self.keys = []
        self.root = None


    def __transplant(self, u, v):

        if u.p is None:
            self.root = v
        elif u == u.p.left:
            u.p.left = v
        else:
            u.p.right = v
        if not (v is None):
            v.p = u.p

        return None


    def __get_min(self, v=None):

        v = self.root if v is None else v

        while (not (v.left is None)):
            v = v.left

        return v


    def __get_max(self, v=None):

        v = self.root if v is None else v

        while (not (v.right is None)):
            v = v.right

        return v


    def __searching(self, key, v=None):

        if v is None or key == v.key:
            return v
        if key < v.key:
            return self.__searching(key, v=v.left)
        else:
            return self.__searching(key, v=v.right)


    def get_minimum(self, v=None):
        return self.__get_min(v=v).key


    def get_maximum(self, v=None):
        return self.__get_max(v=v).key


    def insert(self, z):

        if not (z.key in self.keys):
            self.storage.append(z)
            self.keys.append(z.key)

            y = None
            x = self.root
            while (not (x is None)):
                y = x
                x = x.left if  z.key < x.key else x.right

            z.p = y

            if y is None:
                self.root = z
            elif z.key < y.key:
                y.left = z
            else:
                y.right = z


    def delete(self, z):

        if z.left is None:
            self.__transplant(z, z.right)
        elif z.right is None:
            self.__transplant(z, z.left)
        else:
            y = self.get_minimum(z.right)
            if y.p != z:
                self.__transplant(y, y.right)
                y.right = z.right
                y.right.p = y
            self.__transplant(z, y)
            y.left = z.left
            y.left.p = y

        self.storage.remove(z)
        self.keys.remove(z.key)


    def search(self, key, v=None):
        v = self.root if v is None else v
        result = self.__searching(key, v=v)
        return not (result is None)


    def __str__(self):
        str_root = f"root - {self.root if self.root is None else self.root.key}"
        str_storage = f"storage - {self.storage}"
        return str_root + "\n" + str_storage

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


class CycleFound(Exception):

    def __init__(self):
        super().__init__()


__all__ = [
    'Dsu',
    'Heap',
    'HashTable',
    'BinNode',
    'BinTree',
    'RedBlackTree',
    'Poly',
    'DegreeIsTooBigException',
    'QuadraticPolynomial',
    'CycleFound',
]
