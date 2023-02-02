from .base_cs import go_to_binary
from math import sin, cos, atan2

# Optional:
# Интерполяция многочленов
# Быстрое преобразование Фурье
# Метод Ньютона
# Методы Монте-Карло

# future:
# Геометрические примитивы
# Выпуклые оболочки


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


def sieve_of_eratosthenes(n:int) -> list:
    """
        This is sieve of eratosthenes
        This algorithm give you list is_prime which len: len(is_prime) = n + 1.
        If i - prime number then is_prime[i] = True

    Complexity: O(nlog(log(n)))

    Args:
        n (int): number - end of interval

    Returns:
        list: list is_prime
    """
    is_prime = [True for i in range(n + 1)]
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, n + 1):
        if is_prime[i]:
            for j in range(2*i, n + 1, i):
                is_prime[j] = False
    return is_prime


# def linear_sieve(n:int) -> list:
#     d = [0 for i in range(n + 1)]
#     p = []
#     for k in range(2, n + 1):
#         if p[k] == 0:
#             d[k] = k
#             p.append(k)
#         for x in p:
#             if x > d[k] or x * d[k] > n:
#                 break
#             d[k * x] = x


class Pvector:

    def __init__(self, x:int = 0, y:int = 0):
        self.x = x
        self.y = y


    def __add__(self, v):
        if isinstance(v, Pvector):
            return self.__class__(self.x + v.x, self.y + v.y)
        raise TypeError('You can get sum only two Pvectors')


    def __sub__(self, v):
        if isinstance(v, Pvector):
            return self.__class__(self.x - v.x, self.y - v.y)
        raise TypeError('You can get sub only two Pvectors')


    def __mul__(self, other):
        if isinstance(other, int):
            return self.__class__(self.x * other, self.y * other)
        if isinstance(other, Pvector):
            return self.x * other.x + self.y * other.y
        raise TypeError('You can multiply only Pvector on Prvector and Pvector on int')


    def __pow__(self, other) -> int:
        if isinstance(other, Pvector):
            return self.x * other.y - self.y * other.x
        raise TypeError('You can get a vector product only by Pvector ** Pvector')


    def angle(self) -> float:
        return atan2(self.y, self.x)


    def angle_between_vectors(self, other) -> float:
        return atan2(self ** other, self * other)


    def lenght(self):
        return (self.x**2 + self.y**2)**(1/2)


    def rotation(self, alpha):
        x = self.x
        y = self.y
        self.x = cos(alpha)*x - sin(alpha)*y
        self.y = sin(alpha)*x + cos(alpha)*y



    def __str__(self):
        # s = 'point' if self.is_point else 'vector'
        return f'({self.x}, {self.y})'

    __repr__ = __str__
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__


class LineSegment:

    def __init__(self, A:Pvector, B:Pvector):
        self.A = A
        self.B = B

    def is_intersect(self, other):

        def direction(p1, p2, p3):
            return (p3 - p1) ** (p2 - p1)

        def on_segment(p1, p2, p3):
            return (min(p1.x, p2.x) <= p3.x <= max(p1.x, p2.x)) and (min(p1.y, p2.y) <= p3.y <= max(p1.y, p2.y))


        p1, p2, p3, p4 = self.A, self.B, other.A, other.B
        d1 = direction(p3, p4, p1)
        d2 = direction(p3, p4, p2)
        d3 = direction(p1, p2, p3)
        d4 = direction(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        elif d1 == 0 and on_segment(p3, p4, p1):
            return True
        elif d2 == 0 and on_segment(p3, p4, p2):
            return True
        elif d3 == 0 and on_segment(p1, p2, p3):
            return True
        elif d4 == 0 and on_segment(p1, p2, p4):
            return True
        else:
            return False


class Triangle:

    def __init__(self, A:Pvector, B:Pvector, C:Pvector):
        self.A = A
        self.B = B
        self.C = C

    def get_area(self):
        return abs((1/2) * ((self.B - self.A) ** (self.C - self.A)))



def get_area_of_polygon_gause_clockwise(lst):
    # lst[i] - Pvector
    n = len(lst)
    sum = 0
    for i in range(n):
        sum += lst[i] ** lst[(i + 1) % n]
    return abs(sum) * (1/2)


def get_area_of_polygon_trapezoid_clockwise(lst):
    n = len(lst)
    sum = 0
    for i in range(n):
        x1 = lst[i].x
        y1 = lst[i].y
        x2 = lst[(i + 1) % n].x
        y2 = lst[(i + 1) % n].y
        sum += (x2 - x1) * (y2 + y1) / 2
    return sum


__all__ = [
    'pow',
    'horners_rule',
    'gcd_euclid',
    'euclid_extended',
    'modular_linear_solver',
    'modular_exponentiation',
    'sieve_of_eratosthenes',
    'Pvector',
    'LineSegment',
    'Triangle',
    'get_area_of_polygon_gause_clockwise',
    'get_area_of_polygon_trapezoid_clockwise'
]
