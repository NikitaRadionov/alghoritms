from .data_structures import Heap

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


__all__ = [
    'bubleSort',
    'bubleSort_optimized',
    'heapsort',
]
