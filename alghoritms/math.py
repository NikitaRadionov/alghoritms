from .base_cs import go_to_binary

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

__all__ = [
    'pow',
    'horners_rule',
    'gcd_euclid',
    'euclid_extended',
    'modular_linear_solver',
    'modular_exponentiation',
    'sieve_of_eratosthenes'
]
