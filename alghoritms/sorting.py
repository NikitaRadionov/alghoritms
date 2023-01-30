from .data_structures import Heap

# Optional:
# цифровая сортировка (поразрядная сортировка)

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


def heapSort(lst:list, reverse:bool=False) -> list:
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


def quickSort(lst:list, reverse:bool=False, copy:bool=True) -> list:
    """
        This is algorithm of quick sorting.
        Ascending sort by default.
        Set reverse=True, if you want descending order

    Complexity: O(nlog(n)) - on average
                O(n^2) - worst

    Args:
        lst (list): unsorted list
        reverse (bool, optional): ascending/descending order. Defaults to False.
        copy (bool, optional): If you don't want geting copy of yours array, set copy=False. Defaults to True.

    Returns:
        list: sorted list
    """

    def partitation_asc(lst:list, p:int, r:int):
        x = lst[r]
        i = p - 1
        for j in range(p, r):
            if lst[j] <= x:
                i = i + 1
                c = lst[i]
                lst[i] = lst[j]
                lst[j] = c
        c =lst[i + 1]
        lst[i + 1] = lst[r]
        lst[r] = c
        return i + 1

    def partitation_desc(lst:list, p:int, r:int):
        x = lst[r]
        i = p - 1
        for j in range(p, r):
            if lst[j] >= x:
                i = i + 1
                c = lst[i]
                lst[i] = lst[j]
                lst[j] = c
        c =lst[i + 1]
        lst[i + 1] = lst[r]
        lst[r] = c
        return i + 1

    def sorting(lst:list, p:int, r:int, f):
        if p < r:
            q = f(lst, p, r)
            sorting(lst, p, q - 1, f)
            sorting(lst, q + 1, r, f)
        if p == 0 and r == len(lst) - 1:
            return lst

    f = partitation_desc if reverse else partitation_asc
    array = lst.copy() if copy else lst
    return sorting(array, 0, len(array) - 1, f)


def countingSort(lst:list, k:int) -> list:
    """
        This is Sorting by counting.
        Use this algorithm when you have such array, that each element in [0, k].
        This algorithm sort your array in ascending order and give you a copy of your array

    Complexity: O(n)

    Args:
        lst (list): unsorted array
        k (int): largest int in array

    Returns:
        list: sorted array
    """

    def sorting(A:list, B:list, k:int):
        C = [0 for i in range(k + 1)]

        for j in range(len(A)):
            C[A[j]] = C[A[j]] + 1

        C[0] = C[0] - 1
        for i in range(1, k + 1):
            C[i] = C[i] + C[i - 1]

        for j in range(len(A) - 1, -1, -1):
            B[C[A[j]]] = A[j]
            C[A[j]] = C[A[j]] - 1
        return B

    B = [0 for i in range(len(lst))]

    return sorting(lst, B, k)


def mergeSort(lst:list, reverse:bool=False) -> list:
    """
        This is merge sorting.
        You can get only copy of your sorted list

    Complexity: O(nlog(n))

    Args:
        lst (list): unsorted list
        reverse (bool, optional): asc/desc order. Set reverse=True if you need descending order. Defaults to False.

    Returns:
        list: sorted list
    """

    def merge_asc(a:list, b:list):
        n = len(a)
        m = len(b)
        i, j = 0, 0
        c = []
        while i < n and j < m:
            if a[i]<b[j]:
                c.append(a[i])
                i += 1
            else:
                c.append(b[j])
                j += 1
        while i < n:
            c.append(a[i])
            i += 1
        while j < m:
            c.append(b[j])
            j += 1
        return c


    def merge_desc(a:list, b:list):
        n = len(a)
        m = len(b)
        i, j = 0, 0
        c = []
        while i < n and j < m:
            if a[i]>b[j]:
                c.append(a[i])
                i += 1
            else:
                c.append(b[j])
                j += 1
        while i < n:
            c.append(a[i])
            i += 1
        while j < m:
            c.append(b[j])
            j += 1
        return c


    def sorting(lst:list, f):
        if len(lst) == 1:
            return lst
        middle = len(lst) // 2
        left = sorting(lst[:middle], f)
        right = sorting(lst[middle:], f)

        return f(left, right)


    f = merge_desc if reverse else merge_asc
    return sorting(lst, f)


def insertSort(lst:list, reverse:bool=False, copy:bool=True) -> list:
    """
        This is algorithm of sorting by inserting.
        You can improve complexity to O(nlogn) if you will use bin search for findind
        position for insert element

    Complexity: O(n^2)

    Args:
        lst (list): unsorted list
        reverse (bool, optional): asc/desc order. Set reverse=True if you need descending order. Defaults to False.
        copy (bool, optional): copy/nocopy. Defaults to True.

    Returns:
        list: sorted list
    """
    n = len(lst)
    a = lst.copy() if copy else lst

    def comparsion(a, b, reverse):
        return a > b if reverse else a < b

    for i in range(1, n):
        for j in range(i, 0, -1):
            if comparsion(a[j], a[j - 1], reverse):
                a[j], a[j - 1] = a[j - 1], a[j]
            else:
                break
    return a


def choiceSort(lst:list, reverse:bool=False, copy:bool=True) -> list:
    """
        This is algorithm of sorting by choice.

    Complexity: O(n^2)

    Args:
        lst (list): unsorted list
        reverse (bool, optional): asc/desc order. Set reverse=True if you need descending order. Defaults to False.
        copy (bool, optional): copy/nocopy. Defaults to True.

    Returns:
        list: sorted list
    """

    def comparsion(a, b, reverse):
        return a > b if reverse else a < b


    n = len(lst)
    a = lst.copy() if copy else lst
    for i in range(n):
        k = i
        m = a[i]
        for j in range(i + 1, n):
            if comparsion(a[j], m, reverse):
                m = a[j]
                k = j
        if k != i:
            a[k], a[i] = a[i], a[k]
    return a


def bucketSort(array:list) -> list:
    """
        This is bucket sort algorithm.
        Use this algorithm if you have evenly distributed values in your array
        And your values in array must be >= 0
        values must be int.
        This algorithm give you a sorted copy of yours input array

    Complexity: O(n) - on evenly distributed values
                O(n^2) - on bad data

    Args:
        array (list): unsorted list

    Returns:
        list: sorted list
    """
    largest = max(array)
    length = len(array)
    size = largest/length

    # Create Buckets
    buckets = [[] for i in range(length)]

    # Bucket Sorting
    for i in range(length):
        index = int(array[i]/size)
        if index != length:
            buckets[index].append(array[i])
        else:
            buckets[length - 1].append(array[i])

    # Sorting Individual Buckets
    for i in range(len(array)):
        buckets[i] = sorted(buckets[i])


    # Flattening the Array
    result = []
    for i in range(length):
        result = result + buckets[i]

    return result


__all__ = [
    'bubleSort',
    'bubleSort_optimized',
    'heapSort',
    'quickSort',
    'countingSort',
    'mergeSort',
    'insertSort',
    'choiceSort',
    'bucketSort'
]
