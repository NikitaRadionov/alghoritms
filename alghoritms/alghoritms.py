import time
# Sorting algorithms
def bubleSort(lst:list, reverse=False, copy=True)->list:
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


def bubleSort_optimized(lst:list, reverse=False, copy=True)->list:
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
def strFind_naive(pattern:str, text:str)->int:
    """This is naive realisation algorithm
       for finding a substring in a string

       Complexity: O((n-m+1)m)

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


# maths algorithms
def horners_rule(a, x):

    return None


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


__all__ = ["bubleSort", "bubleSort_optimized", "strFind_naive"]

if __name__ == "__main__":
    print("Hi")
    a = Poly(1,2,3,4)
    print(a)