import time

def bubleSort(lst:list, reverse=False, copy=True)->list:
    """
        Buble sort. Naive realisation. Complexity O(n^2)
        Ascending sort by default.
        Set reverse=True, if you want descending order
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
        Buble sort. Optimized realisation. Complexity O(n^2)
        Ascending sort by default.
        Set reverse=True, if you want descending order
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

__all__ = ["bubleSort"]

if __name__ == "__main__":
    a = [2,1,5,3,9,10,4,5,3,2,1,5,8,23,7,0,54,21,43,12,23,1,2,3,4,9,5,2,1,5,2,0,32,54,1,45,78,12,32,93,76,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,11,1,1,11,1,1,1,11,1,11,1,1,11,1,11,1,11,1,11,11,1]
