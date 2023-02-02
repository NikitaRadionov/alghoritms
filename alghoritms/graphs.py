from typing import Any
from .data_structures import Dsu, CycleFound
from queue import Queue, PriorityQueue

# Optional:
# modern Euler cycle

# future:
# Bridges and articulation points
# Компоненты сильной связности
# Пути в ациклических графах
# Корневые деревья
# Dinic
# mincost flow
# Floyd


def listadj_to_listedge(lst:list | dict) -> list:
    """
        This algorithm for creating list of edges by list of adjacency for unweight graph.

    Complexity: O(m) where m - count of edges in graph

    Args:
        lst (list | dict): list of adjacency

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list: list of edges
    """
    if isinstance(lst, (list, dict)):
        vertexes = range(len(lst)) if isinstance(lst, list) else lst.keys()
        edges = []
        for u in vertexes:
            for v in lst[u]:
                edges.append((u, v))
        return edges
    else:
        raise TypeError('Wrong Argumnets')


def listadj_weight_to_listedge(lst:list | dict) -> list:
    """
        This algorithm for creating list of edges by list of adjacency for weight graph.

    Complexity: O(m) where m - count of edges in graph

    Args:
        lst (list | dict): list of adjacency

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list: list of edges
    """
    if isinstance(lst, (list, dict)):
        vertexes = range(len(lst)) if isinstance(lst, list) else lst.keys()
        edges = []
        for u in vertexes:
            for v in lst[u]:
                edges.append((u, v[0], v[1]))
        return edges
    else:
        raise TypeError('Wrong Argumnets')


def listadj_to_matrixadj(lst:list | dict) -> list | dict:
    """
        This algorithm for creating matrix adjacency by list of adjacency for unweight graph.
        If edge (a, b) is exist, I suggest, that matrix[a][b] == 1
        If edge (a, b) does not exist, I suggest, that matrix[b][b] == 0

    Complexity: O(n^2) where n - count of vertexes in graph

    Args:
        lst (list | dict): list of adjacency

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list | dict: matrix of adjacency
    """
    if isinstance(lst, (list, dict)):

        all_vertexes = list(range(len(lst))) if isinstance(lst, list) else list(lst.keys())
        vertexes = all_vertexes.copy()

        for u in vertexes:
            for v in lst[u]:
                if not (v in all_vertexes):
                    all_vertexes.append(v)

        if isinstance(lst, list):
            n = len(all_vertexes)
            matrix = [[0 for j in range(n + 1)] for i in range(n + 1)]

            for a in all_vertexes:
                for b in lst[a]:
                    matrix[a][b] = 1

            return matrix
        else:

            matrix = {v: {v: 0 for v in all_vertexes} for v in all_vertexes}

            for a in all_vertexes:
                if a in lst.keys():
                    for b in lst[a]:
                        matrix[a][b] = 1

            return matrix

    else:
        raise TypeError('Wrong Argumnets')


def listadj_weight_to_matrixadj(lst:list | dict) -> list | dict:
    """
        This algorithm for creating matrix adjacency by list of adjacency for weight graph.
        If edge (a, b, w) is exist, I suggest, that matrix[a][b] == w
        If edge (a, b, w) does not exist, I suggest, that matrix[b][b] == -1

    Complexity: O(n^2) where n - count of vertexes in graph

    Args:
        lst (list | dict): list of adjacency

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list | dict: matrix of adjacency
    """
    if isinstance(lst, (list, dict)):

        all_vertexes = list(range(len(lst))) if isinstance(lst, list) else list(lst.keys())
        vertexes = all_vertexes.copy()

        for u in vertexes:
            for v in lst[u]:
                if not (v[0] in all_vertexes):
                    all_vertexes.append(v[0])

        if isinstance(lst, list):
            n = len(all_vertexes)
            matrix = [[0 for j in range(n + 1)] for i in range(n + 1)]

            for a in all_vertexes:
                for b in lst[a]:
                    matrix[a][b[0]] = b[1]

            return matrix
        else:

            matrix = {v: {v: -1 for v in all_vertexes} for v in all_vertexes}

            for a in all_vertexes:
                if a in lst.keys():
                    for b in lst[a]:
                        matrix[a][b[0]] = b[1]

            return matrix
    else:
        raise TypeError('Wrong Argumnets')


def listedge_to_listadj(lst:list, numbers:bool=True) -> list | dict:
    """
        This algorithm for create a list of adjancency by list of edges.
        Edges in list of edges a must be a tuples of this form (a, b).
        (a, b) means that the edge starts in vertex a and ends in vertex b.
        By default I suggest that a and b is int such that a,b >= 0.
        If a and b is not int set numbers=False and you get a dict of adjancency.
        If you set numbers=False then I suggest that a,b is not mutable
        and using this function you will get an adjacency dictionary.
        I suggest that you are using this algorithm for undirected unweighted graph but
        this also works for directed unweighted graph

    Complexity: O(m) where m - count of edges in graph

    Args:
        a (list): list of edges. Edges must be a tuples of this form (a, b).
        numbers (bool, optional): set False if a or b is not int such that a,b >= 0. Defaults to True.

    Returns:
        list | dict: list of adjancency | dict of adjancency
    """
    if numbers:
        n = 0
        for (a, b) in lst:
            mx = max(a, b)
            if mx > n:
                n = mx
        adjlst = [[] for i in range(n + 1)]
        for (a, b) in lst:
            adjlst[a].append(b)
        return adjlst
    else:
        d = {}
        for (a, b) in lst:
            if not (a in d.keys()):
                d[a] = []
            d[a].append(b)
        return d


def listedge_to_matrixadj(lst:list, numbers:bool=True) -> list | dict:
    """
        This algorithm for create a matrix of adjancency by list of edges.
        Edges in list of edges a must be a tuples of this form (a, b).
        (a, b) means that the edge starts in vertex a and ends in vertex b.
        By default I suggest that a and b is int such that a,b >= 0.
        If a and b is not int set numbers=False and you get a dict of adjancency.
        If you set numbers=False then I suggest that a,b is not mutable
        and using this function you will get an adjacency dictionary.
        I suggest that you are using this algorithm for undirected unweighted graph but
        this also works for directed unweighted graph

    Complexity: O(m^2) where m - count of edges

    Args:
        lst (list): list of edges. Edges must be a tuples of this form (a, b).
        numbers (bool, optional): set False if a or b is not int such that a,b >= 0. Defaults to True.

    Returns:
        list | dict: matrix of adjancency | matrix-dict of adjancency
    """
    if numbers:
        n = 0
        for (a, b) in lst:
            mx = max(a, b)
            if mx > n:
                n = mx
        matrix = [[0 for j in range(n + 1)] for i in range(n + 1)]
        for (a, b) in lst:
            matrix[a][b] = 1
        return matrix
    else:
        vertexes = []
        for edge in lst:
            for i in range(2):
                if not (edge[i] in vertexes):
                    vertexes.append(edge[i])

        d = {v: {v: 0 for v in vertexes} for v in vertexes}
        for (a, b) in lst:
            d[a][b] = 1
        return d


def listedge_weight_to_listadj(lst:list, numbers:bool=True) -> list | dict:
    """
        This algorithm for create a list of adjancency by list of edges.
        Edges in list of edges a must be a tuples of this form (a, b, w).
        (a, b, w) means that the edge starts in vertex a and ends in vertex b and edge have weight w.
        By default I suggest that a and b is int such that a,b >= 0.
        If a and b is not int set numbers=False and you get a dict of adjancency.
        If you set numbers=False then I suggest that a,b is not mutable
        and using this function you will get an adjacency dictionary.
        I suggest that you are using this algorithm for undirected unweighted graph but
        this also works for directed weighted graph

    Complexity: O(m) where m is count of edges

    Args:
        lst (list): list of edges. Edges must be a tuples of this form (a, b, w). w - weight
        numbers (bool, optional): set False if a or b is not int such that a,b >= 0. Defaults to True.

    Returns:
        list | dict: list of adjancency | dict of adjancency
    """
    if numbers:
        n = 0
        for (a, b, w) in lst:
            mx = max(a, b)
            if mx > n:
                n = mx
        adjlst = [[] for i in range(n + 1)]
        for (a, b, w) in lst:
            adjlst[a].append((b, w))
        return adjlst
    else:
        d = {}
        for (a, b, w) in lst:
            if not (a in d.keys()):
                d[a] = []
            d[a].append((b, w))
        return d


def listedge_weight_to_matrixadj(lst:list, numbers:bool=True) -> list | dict:
    """
        This algorithm for create a matrix of adjancency by list of edges.
        Edges in list of edges a must be a tuples of this form (a, b, w).
        (a, b, w) means that the edge starts in vertex a and ends in vertex b and edge have weight w..
        By default I suggest that a and b is int such that a,b >= 0.
        If a and b is not int set numbers=False and you get a dict of adjancency.
        If you set numbers=False then I suggest that a,b is not mutable
        and using this function you will get an adjacency dictionary.
        I suggest that you are using this algorithm for undirected unweighted graph but
        this also works for directed weighted graph

    Complexity: O(m^2) where m - count of edges

    Args:
        lst (list): list of edges. Edges must be a tuples of this form (a, b, w).
        numbers (bool, optional): set False if a or b is not int such that a,b >= 0. Defaults to True.

    Returns:
        list | dict: matrix of adjancency | matrix-dict of adjancency
    """
    if numbers:
        n = 0
        for (a, b, w) in lst:
            mx = max(a, b)
            if mx > n:
                n = mx
        matrix = [[-1 for j in range(n + 1)] for i in range(n + 1)]
        for (a, b, w) in lst:
            matrix[a][b] = w
        return matrix
    else:

        vertexes = []
        for edge in lst:
            for i in range(2):
                if not (edge[i] in vertexes):
                    vertexes.append(edge[i])

        d = {v: {v: -1 for v in vertexes} for v in vertexes}
        for (a, b, w) in lst:
            d[a][b] = w
        return d


def matrixadj_to_listedge(matrix:list | dict) -> list:
    """
        This is algorithm for creating list of edges by matrix adjacency for unweight graph.
        If edge (a, b) does not exist, then I suggest that matrix[a][b] == 0
        If edge (a, b) is exist, then I suggest that matrix[a][b] == 1
        a, b may be any unmutable objects

    Complexity: O(n^2) where n - count of vertexes in graph

    Args:
        matrix (list | dict): matrix of adjacency

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list: list of edges
    """
    if isinstance(matrix, (list, dict)):
        vertexes = range(len(matrix)) if isinstance(matrix, list) else matrix.keys()
        lst = []
        for a in vertexes:
            for b in vertexes:
                if matrix[a][b] == 1:
                    lst.append((a, b))
        return lst
    else:
        raise TypeError('Wrong arguments')


def matrixadj_weight_to_listedge(matrix:list | dict) -> list:
    """
        This is algorithm for creating list of edges by matrix adjacency for weight graph.
        If edge (a, b, w) does not exist, then I suggest that matrix[a][b] == -1
        If edge (a, b, w) is exist, then I suggest that matrix[a][b] == w
        a, b may be any unmutable objects

    Complexity: O(n^2) where n - count of vertexes in graph

    Args:
        matrix (list | dict): matrix of adjacency

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list: list of edges
    """
    if isinstance(matrix, (list, dict)):
        vertexes = range(len(matrix)) if isinstance(matrix, list) else matrix.keys()
        lst = []
        for a in vertexes:
            for b in vertexes:
                if matrix[a][b] != -1:
                    lst.append((a, b, matrix[a][b]))
        return lst
    else:
        raise TypeError('Wrong arguments')


def matrixadj_to_listadj(matrix:list| dict) -> list:
    """
        This algorithm for creating list of adjacency by matrix of adjacency for unweight graph.
        If edge (a, b) is exist, I suggest, that matrix[a][b] == 1
        If edge (a, b) does not exist, I suggest, that matrix[a][b] == 0

    Complexity: O(n^2) where n - count of edges in graph

    Args:
        matrix (list | dict): matrix of adjacency

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list: list of adjacency
    """
    if isinstance(matrix, (list, dict)):
        vertexes = range(len(matrix)) if isinstance(matrix, list) else matrix.keys()
        lst = [[] for i in range(len(vertexes))] if isinstance(matrix, list) else {v: [] for v in vertexes}
        for a in vertexes:
            for b in vertexes:
                if matrix[a][b] == 1:
                    lst[a].append(b)
        return lst
    else:
        raise TypeError('Wrong arguments')


def matrixadj_weight_to_listadj(matrix:list| dict) -> list:
    """
        This algorithm for creating list of adjacency by matrix of adjacency for weight graph.
        If edge (a, b, w) is exist, I suggest, that matrix[a][b] == w
        If edge (a, b, w) does not exist, I suggest, that matrix[a][b] == -1

    Complexity: O(n^2) where n - count of edges in graph

    Args:
        matrix (list | dict): matrix of adjacency

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list: list of adjacency
    """
    if isinstance(matrix, (list, dict)):
        vertexes = range(len(matrix)) if isinstance(matrix, list) else matrix.keys()
        lst = [[] for i in range(len(vertexes))] if isinstance(matrix, list) else {v: [] for v in vertexes}
        for a in vertexes:
            for b in vertexes:
                if matrix[a][b] != -1:
                    lst[a].append((b, matrix[a][b]))
        return lst
    else:
        raise TypeError('Wrong arguments')


def get_empty_color(lst:list | dict) -> list | dict:
    """
        This is helper function for dfs algorithms.
        With her help dfs algorithms gets an empty list | dict of vertex colors.

    Complexity: O(n) where n - count of vertex in graph

    Args:
        lst (list | dict): list/dict of adjacency

    Raises:
        TypeError: Wrong arguments

    Returns:
        list | dict: list/dict of vertex colors
    """

    if isinstance(lst, list):
        n = len(lst)
        color = [0 for i in range(n)]
        return color

    if isinstance(lst, dict):
        color = {}
        for obj in lst.keys():
            color[obj] = 0
        return color

    raise TypeError("Wrong arguments")


def graph_dfs(lst:list | dict, color:list | dict, u:Any, weight:bool = False):
    """
        This is algorithm of depth-walk search which start from vertex of undirected/directed unweight graph.
        I suggest that you are using for this algorithm a list/dict of adjacency which you
        can create by a function get_graph_listadjacency.
        color[i] must be one of this int - 0, 1, 2.
        color must be the same type as lst.
        u must be not mutable

    Complexity: O(m) where m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        color (list | dict): color of vertex
        u (Any): vertex from which dfs is statring
        weight (bool): set weight = True if your graph are weighted. Defaults to False.
    """
    if weight:
        color[u] = 1
        for (v, w) in lst[u]:
            if color[v] == 0:
                graph_dfs(lst, color, v, weight)
        color[u] = 2
    else:
        color[u] = 1
        for v in lst[u]:
            if color[v] == 0:
                graph_dfs(lst, color, v)
        color[u] = 2


def graph_dodfs(lst:list | dict, weight:bool = False) -> list | dict:
    """
        This is algorithm of depth-walk search on undirected/directed unweight graph.
        I suggest that you are using for this algorithm a list/dict of adjacency which you
        can create by a function get_graph_listadjacency.

    Complexity: O(n + m)
                where n - count of vertex in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        weight (bool): set weight = True if your graph are weighted. Defaults to False.

    Raises:
        TypeError: Wrong arguments

    Returns:
        list | dict: list/dict of colors - the result of walking
    """
    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    color = get_empty_color(lst)
    array = range(len(lst)) if condition_list else lst.keys()

    for u in array:
        graph_dfs(lst, color, u, weight)

    return color


def get_all_connected_components(lst: list| dict, weight:bool = False) -> tuple:
    """
        This algorithm find all connected components in your weighted/unweighted graph.
        This algorithm returns a tuple of this form: (component, count)
        where component: list | dict such that component[v] = number connected component.
              count - count connected components in your graph
        This algorithm based on dfs

    Complexity: O(n + m)
                where n - count of vertexes in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        weight (bool, optional): set True if you have weighted graph. Defaults to False.

    Raises:
        TypeError: Wrong arguments

    Returns:
        tuple: (component, count)
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    def advanced_dfs(v:Any, num:int, lst: list | dict, component:list | dict, weight:bool = False):
        component[v] = num
        if weight:
            for (u, w) in lst[v]:
                if component[u] == 0:
                    advanced_dfs(u, num, lst, component, weight)
        else:
            for u in lst[v]:
                if component[u] == 0:
                    advanced_dfs(u, num, lst, component, weight)


    array = range(len(lst)) if condition_list else lst.keys()
    component = [0 for i in range(len(lst))] if condition_list else {obj: 0 for obj in lst.keys()}

    num = 0

    for v in array:
        if component[v] == 0:
            num += 1
            advanced_dfs(v, num, lst, component, weight)

    return (component, num)


def is_bipartite(lst: list|dict, weight:bool = False) -> tuple | bool:
    """
        This is algorithm for determine bipartite of graph.
        If your graph is bipartite then you get tuple of this form: (True, color)
        True - bool that tell you graph is bipartite
        color - list | dict - this is coloring book for each vertex in your graph
        This algorithm based on dfs

    Complexity: O(n + m)
                where n - count of vertexes in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        weight (bool, optional): Set True if your graph are weighted. Defaults to False.

    Raises:
        TypeError: Wrong arguments

    Returns:
        tuple | bool: (True, color) | False
    """
    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    def advanced_dfs(v:Any, col:int, lst: list | dict, color:list | dict, weight:bool = False):
        color[v] = col
        if weight:
            for (u, w) in lst[v]:
                if color[u] == 0:
                    result = advanced_dfs(u, -col, lst, color, weight)
                    if isinstance(result, bool):
                        return False
                elif color[u] != - col:
                    return False
        else:
            for u in lst[v]:
                if color[u] == 0:
                    result = advanced_dfs(u, -col, lst, color, weight)
                    if isinstance(result, bool):
                        return False
                elif color[u] != - col:
                    return False

    array = range(len(lst)) if condition_list else lst.keys()
    color = [0 for i in range(len(lst))] if condition_list else {obj: 0 for obj in lst.keys()}

    for v in array:
        if color[v] == 0:
            result = advanced_dfs(v, 1, lst, color, weight)
            if isinstance(result, bool):
                return False

    return (True, color)


def have_cycle(lst: list|dict, weight:bool = False) -> tuple:
    """
        This algorithm find one cycle in graph.
        Use this algorithm if you need to determine a tree
        This algorithm based of dfs

    Complexity: O(n + m)
                where n - count of vertexes in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        weight (bool, optional): Set True if your graph are weighted. Defaults to False.

    Raises:
        TypeError: Wrong Arguments

    Returns:
        tuple: (bool, cycle)
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    def alternative_dfs(v:Any,used:list | dict, lst: list | dict, p:Any = -1, weight:bool=False, flg:lst=[]) -> list:
        if used[v]:
            s = 'cycle: '
            s += str(v)
            return [True, s, v]

        used[v] = 1

        if weight:
            for (u, w) in lst[v]:
                try:
                    if u != p:
                        k = alternative_dfs(u, used, lst, p=v, weight=weight, flg=flg)
                        if len(flg) == 0 and k[2] != -1:
                            k[1] += ' <- ' + str(v)
                            if k[2] == v:
                                # exit(0)
                                flg.append(1)
                                raise CycleFound
                            return k
                        else:
                            return k
                except CycleFound:
                    return k
        else:
            for u in lst[v]:
                try:
                    if u != p:
                        k = alternative_dfs(u,used, lst, p=v ,weight=weight, flg=flg)
                        if len(flg) == 0 and k[2] != -1:
                            k[1] += ' <- ' + str(v)
                            if k[2] == v:
                                # exit(0)
                                flg.append(1)
                                raise CycleFound
                            return k
                        else:
                            return k
                except CycleFound:
                    return k

        return [False, '', -1]


    array = range(len(lst)) if condition_list else lst.keys()
    used = [0 for i in range(len(lst))] if condition_list else {obj: 0 for obj in lst.keys()}

    for v in array:
        if used[v] == 0:
            k = alternative_dfs(v, used, lst, weight=weight)
            if k[0]:
                return (k[0], k[1])
    return (False, '')


def topological_sort(lst: list | dict, weight:bool = False) -> list:
    """
        This is algorithm of topological sorting of graph
        Whis algorithm works for acyclic graph
        Algorithm returns list t - topological order of vertexes in graph.
        topological orders means:
        Let t[i] = u, t[j] = v.
        for each i,j: i < j we have no path from v to u

        This algorithm baased on dfs

    Complexity: O(n + m)
                where n - count of vertexes in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency
        weight (bool, optional): Set True if your graph are weighted. Defaults to False.

    Raises:
        TypeError: Wrong arguments

    Returns:
        list: topological order
    """
    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    def dfs(v:Any, lst: list | dict, used: list | dict, t:list):
        used[v] = 1
        if weight:
            for (u, w) in lst[v]:
                if used[u] == 0:
                    dfs(u, lst, used, t)
            t.append(v)
        else:
            for u in lst[v]:
                if used[u] == 0:
                    dfs(u, lst, used, t)
            t.append(v)

    array = range(len(lst)) if condition_list else lst.keys()
    used = [0 for i in range(len(lst))] if condition_list else {obj: 0 for obj in lst.keys()}
    t = []

    for v in array:
        if used[v] == 0:
            dfs(v, lst, used, t)
    t.reverse()
    return t


def graph_bfs(lst:list | dict, s:Any, weight:bool = False) -> tuple:
    """
        This algorithm of bfs on graph.
        Input graph must be unweighted
        This algorithm returns tuple of this form (visited, d, p)
        visited - array of visited vertexes
        d - array of shortest paths from vertex s. d[v] - shortest path from s to v
        p - array of parrents. p[v] - parrent of vertex v

    Complexity: O(n + m)
                where n - count of vertex
                      m - count of edges

    Args:
        lst (list | dict): list/dict of adjacency
        s (Any): Start vertex
        weight (bool, optional): set True, if you want to get expirement. Defaults to False.

    Returns:
        tuple: (visited, d, p)
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrond arguments')

    q = Queue()
    q.put(s)

    if condition_list:
        n = len(lst)
        visited = [False for i in range(n)]
        p = [None for i in range(n)]
        d = [-1 for i in range(n)]
    else:
        visited, p, d = {}, {}, {}
        for obj in lst.keys():
            visited[obj] = False
            p[obj] = None
            d[obj] = -1

    visited[s] = True
    d[s] = 0

    while not (q.empty()):
        v = q.get()
        if weight:
            for (u, w) in lst[v]:
                if not visited[u]:
                    q.put(u)
                    visited[u] = True
                    d[u] = d[v] + w
                    p[u] = v
        else:
            for u in lst[v]:
                if not visited[u]:
                    q.put(u)
                    visited[u] = True
                    d[u] = d[v] + 1
                    p[u] = v

    return (visited, d, p)


def dejcstra(lst:list | dict, s:Any, max_weight:Any = None) -> list | dict:
    """
        This is Dejcsta's algorithm for findind shortest paths from s to other vertexes in weighted graph.
        Attention: weight of each edge must be >= 0.
        If you know the maximal weight of edge in graph (let it be max), set max_weight = max + 1

        Algotihm:
            Start:
                d[s] = 0 , a[v] = 0
                for each v in V\{s}: d[v] = infinity
             Basis:
                1. Choose v:
                    v = min{d[u]: a[u] == 0}
                2. a[v] = 1
                3. decompression:
                    1. for each (v, u, w):
                        d[u] = min(d[u], d[v] + w)

    Complexity: O(n^2) where n - count vertexes in graph

    Args:
        lst (list | dict): list/dict of adjacency
        s (Any): start vertex. s is not mutable
        max_weight (Any, optional): max weight + 1. Defaults to None.

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list | dict: array of minimal weight paths from vertex s to other vertexes in graph
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    max_weight = max_weight if not ( max_weight is None ) else 1.8446744e+19
    n = len(lst)

    a = [False for i in range(n)] if condition_list else {obj: 0 for obj in lst.keys()}
    d = [max_weight for i in range(n)] if condition_list else {obj: max_weight for obj in lst.keys()}

    # if condition_list:
    #     a = [False for i in range(n)]
    #     d = [max_weight for i in range(n)]
    # else:
    #     a, d = {}, {}
    #     for i in lst.keys():
    #         a[i] = 0
    #         d[i] = max_weight

    d[s] = 0

    array = range(n) if condition_list else lst.keys()
    for i in range(n):
        v = -1 if condition_list else None
        for u in array:
            if (not a[u]) and ( (v in (-1, None)) or d[u] < d[v]):
                v = u

        a[v] = True

        for (u, w) in lst[v]:
            d[u] = min(d[u], d[v] + w)

    return d


def dejcstra_heap(lst:list | dict, s:Any, max_weight:Any = None) -> list | dict:
    """
        This is Dejcsta's algorithm for findind shortest paths from s to other vertexes in weighted graph.
        Attention: weight of each edge must be >= 0.
        If you know the maximal weight of edge in graph (let it be max), set max_weight = max + 1

        This is improved version of this algorithm.
        For finding v = min{d[u]: a[u] == 0} is using priority queue which based on heap

    Complexity: O(nlogn + m)
                where n - count of vertex in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): _description_
        s (Any): _description_
        max_weight (Any, optional): _description_. Defaults to None.

    Raises:
        TypeError: _description_

    Returns:
        list | dict: _description_
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    max_weight = max_weight if not ( max_weight is None ) else 1.8446744e+19
    n = len(lst)

    if condition_list:
        d = [max_weight for i in range(n)]

    else:
        d = {}
        for i in lst.keys():
            d[i] = max_weight

    d[s] = 0

    q = PriorityQueue()
    q.put((0, s))
    while not q.empty():
        (cur_d, v) = q.get()
        if cur_d > d[v]:
            continue
        for (u, w) in lst[v]:
            if d[u] > d[v] + w:
                d[u] = d[v] + w
                q.put((d[u], u))

    return d


def prims_algorithm_naive(lst:list) -> list:
    """
        This is naive realisation of Prim's algorithm for finding minimal island.
        Minimal island is such spanning tree of graph that sum of edge weights is minimal
        Input: Connected weighted undirected graph
        Output: Minimal island (list of edges)

    Complexity: O(nm)
                where n - count of vertexes in graph
                      m - count of edges in graph

    Args:
        lst (list): list of edges

    Raises:
        TypeError: Wrong arguments

    Returns:
        list: list of edges of minimal island
    """

    condition_list = isinstance(lst, list)
    condition_element = isinstance(lst[0], tuple)
    condition_numbers = isinstance(lst[0][0], int)

    if not ( condition_list and condition_element):
        raise TypeError('Wrong arguments')

    adj_lst = listedge_weight_to_listadj(lst, numbers=condition_numbers)

    inf = 1.8446744e+19
    n = len(adj_lst)
    m = len(lst)
    island = []
    used = [0 for i in range(n)] if condition_numbers else {obj: 0 for obj in adj_lst.keys()}
    element = 0 if condition_numbers else lst[0][0]

    used[element] = 1
    for i in range(n - 1):
        isl_w, isl_a, isl_b = inf, -1, -1
        for j in range(m):
            a, b, w = lst[j]
            if isl_w > w and used[a] != 0 and used[b] == 0:
                isl_w = w
                isl_a = a
                isl_b = b
        used[isl_b] = 1
        island.append((isl_a, isl_b, isl_w))
        island.append((isl_b, isl_a, isl_w))

    return island


def prims_algorithm_optimized_square(lst:list | dict) -> list:
    """
        This is optimized realisation of prims algorithm for finding minimal island.
        Minimal island is such spanning tree of graph that sum of edge weights is minimal
        Input: Connected weighted undirected graph
        Output: Minimal island

        Use this algorithm if you have a dense graph

    Complexity: O(n^2)
                where n - count of vertexes in graph

    Args:
        lst (list | dict): list/dict of adjacency

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list: list of edges of minimal island
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)

    if not (condition_list or condition_dict):
        raise TypeError('Wrong Arguments')


    inf = 1.8446744e+19
    n = len(lst)
    island = []
    used = [0 for i in range(n)] if condition_list else {obj: 0 for obj in lst.keys()}
    min_edge = [inf for i in range(n)] if condition_list else {obj: inf for obj in lst.keys()}
    start = 0 if condition_list else next(iter(lst.keys()))
    min_edge[start] = 0
    best_edge = used.copy()

    array = range(n) if condition_list else lst.keys()

    for i in range(n):
        v = -1
        for u in array:
            if used[u] == 0 and (v == -1 or min_edge[u] < min_edge[v]):
                v = u

        used[v] = 1

        if v != start:
            island.append((best_edge[v], v, min_edge[v]))
            island.append((v, best_edge[v], min_edge[v]))

        for edge in lst[v]:
            u = edge[0]
            w = edge[1]
            if w < min_edge[u]:
                min_edge[u] = w
                best_edge[u] = v

    return island


def prims_algorithm_optimized_log(lst: list | dict) -> list:
    """
        This if optimized realisation of Prim's algorithm for finding minimal island.
        Minimal island is such spanning tree of graph that sum of edge weights is minimal
        Input: Connected weighted undirected graph
        Output: Minimal island (list of edges)

        In this realisation is used priority queue for finding minimal weight edge in each step

        Use this algorithm if you have a sparse graph

    Complexity: O(mlogn)
                where n - count of vertexes in graph
                      m - count of edges in graph

    Args:
        lst (list | dict): list/dict of adjacency

    Raises:
        TypeError: Wrong Arguments

    Returns:
        list: list of edges of minimal island
    """

    condition_list = isinstance(lst, list)
    condition_dict = isinstance(lst, dict)


    if not ( condition_list or condition_dict):
        raise TypeError('Wrong arguments')

    n = len(lst)

    island = []
    start = 0 if condition_list else next(iter(lst.keys()))
    U = {start}
    q = PriorityQueue()

    for (u, w) in lst[start]:
        q.put((w, (start, u)))

    while len(U) < n:
        isl_edge = q.get()
        v = isl_edge[1][0]
        u = isl_edge[1][1]
        w = isl_edge[0]

        if (not ((v, u, w) in island)) and (v in U and (not (u in U))):
            U.add(u)
            island.append((v, u, w))
            island.append((u, v, w))

        for (v, w) in lst[u]:
            if (u in U and (not (v in U))) and (not ((w, (u, v)) in q.queue)):
                q.put((w, (u, v)))

    return island


def kruskals_algorithm(lst:list) -> list:
    """
        This is Kruskal's algorithm for findind minimal island.
        Minimal island is such spanning tree of graph that sum of edge weights is minimal
        Input: Connected weighted undirected graph (list of edges)
        Output: Minimal island (list of edges)

        This implementation uses disjoin set union

    Complexity: O(mlog(m))
                where m - count of edges in graph

    Args:
        lst (list): list of edges of your graph

    Returns:
        list: list of edges of minimal island
    """

    sorted_lst = sorted(lst, key=lambda x: x[2])
    island = []
    s = Dsu()

    # initialization disjoin set union
    for edge in lst:
        a = edge[0]
        b = edge[1]
        if not s.have_element(a):
            s.make_set(a)
        if not s.have_element(b):
            s.make_set(b)

    # algorithm
    for edge in sorted_lst:
        a = edge[0]
        b = edge[1]
        w = edge[2]
        if s.find_set(a) != s.find_set(b):
            island.append((a, b, w))
            island.append((b, a, w))
            s.union_sets(a, b)

    return island


def get_euler_cycle(list_of_edges:list, numbers=True) -> str:
    """
        This is algorithm for finding Euler cycle in unordered unweight graph.
        This algorithm returns you right answer if your graph is euler graph


    Complexity: O(n + m) where n - count of vertexes
                               m - count of edges

    Args:
        lst (list): list of edges
    Returns:
        str: Euler cycle
    """

    def euler(u, lst, visited, first, cycle):
        while (first[u] < len(lst[u])):
            p = lst[u][first[u]]
            i = p[0]
            v = p[1]
            if not visited[i]:
                visited[i] = 1
                euler(v, lst, visited, first, cycle)
                cycle.append((v, u))
                # print(v, u)
            first[u] += 1


    # creating list of adjacency like this:
    # lst = {
    #    'a': [(i, 'b'), (j, 'c'), ...]
    #     ...
    # }
    # i - serial number of edge (a, b)
    def get_work_list(list_of_edges, numbers):
        work_flow = {}
        i = -1
        for edge in list_of_edges:
            a = edge[0]
            b = edge[1]
            if str((b, a)) in list(work_flow.keys()):
                work_flow[str(edge)] = work_flow[str((b, a))]
            else:
                i += 1
                work_flow[str(edge)] = i
        if numbers:
            n = 0
            for (a, b) in list_of_edges:
                mx = max(a, b)
                if mx > n:
                    n = mx
            adjlst = [[] for i in range(n + 1)]
            for (a, b) in list_of_edges:
                adjlst[a].append((work_flow[str((a, b))], b))
            return adjlst
        else:
            d = {}
            for (a, b) in list_of_edges:
                if not (a in d.keys()):
                    d[a] = []
                d[a].append((work_flow[str((a, b))], b))
            return d


    lst = get_work_list(list_of_edges, numbers)
    first = [0 for i in range(len(lst))] if numbers else {v: 0 for v in lst.keys()}
    visited = [0 for i in range(len(list_of_edges))]
    start = list_of_edges[0][0]
    cycle =[]
    euler(start, lst, visited, first, cycle)
    string = ""
    for i in range(len(cycle) - 2, - 2, -1):
        string += ' -> ' + str(cycle[i + 1][0])
    return str(start) + string


def get_bridge(lst: list | dict):

    def dfs(v, visited, d, h, bridges, p = -1):
        visited[v] = 1
        compute = 0 if p == -1 else h[p] + 1
        d[v] = compute
        h[v] = compute
        for u in lst[v]:
            if u != p:
                if visited[u]:
                    d[v] = min(d[v], h[u])
                else:
                    dfs(u, visited, d, h, bridges, p=v)
                    d[v] = min(d[v], d[u])
                    if h[v] < d[u]:
                        bridges.append((v, u))
                        # print(v, u)

    visited = [0 for i in range(len(lst))] if isinstance(lst, list) else {v: 0 for v in lst.keys()}
    d, h = {}, {}
    start = 0 if isinstance(lst, list) else list(lst.keys())[0]
    bridges = []
    dfs(start, visited, d, h, bridges)
    return bridges



__all__ = [
    'listadj_to_listedge',
    'listadj_weight_to_listedge',
    'listadj_to_matrixadj',
    'listadj_weight_to_matrixadj',
    'listedge_to_listadj',
    'listedge_to_matrixadj',
    'listedge_weight_to_listadj',
    'listedge_weight_to_matrixadj',
    'matrixadj_to_listedge',
    'matrixadj_weight_to_listedge',
    'matrixadj_to_listadj',
    'matrixadj_weight_to_listadj',
    'get_empty_color',
    'graph_dfs',
    'graph_dodfs',
    'get_all_connected_components',
    'is_bipartite',
    'have_cycle',
    'topological_sort',
    'graph_bfs',
    'dejcstra',
    'dejcstra_heap',
    'prims_algorithm_naive',
    'prims_algorithm_optimized_square',
    'prims_algorithm_optimized_log',
    'kruskals_algorithm',
    'get_euler_cycle',
    'get_bridge'
]
