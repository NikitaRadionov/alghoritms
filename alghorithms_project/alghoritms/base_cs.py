from typing import Any

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


__all__ = [
    'go_to_binary',
    'linear_search',
    'binary_search',
    'binary_search_imaginary'
]
