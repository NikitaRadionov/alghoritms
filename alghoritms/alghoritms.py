import time

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


__all__ = ["bubleSort", "bubleSort_optimized", "strFind_naive"]

if __name__ == "__main__":
    print("Hi")