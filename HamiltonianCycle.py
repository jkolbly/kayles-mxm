import Graphs
import networkx as nx
from pysat.solvers import Glucose3

def hamCycSolver(G: nx.Graph) -> Glucose3:
    G.remove_nodes_from([n for n in G if G.degree[n] == 0])

    solver = Glucose3()

    n = len(G)
    x = lambda i,j: i * n + j + 1

    # Ensure all nodes are in path:
    for j in range(n):
        solver.add_clause([x(i,j) for i in range(n)])

    # Ensure no nodes repeat in path
    for j in range(n):
        for i in range(n):
            for k in range(n):
                if i == k:
                    continue
                solver.add_clause([-x(i,j), -x(k,j)])

    # Ensure all slots in path are occupied
    for i in range(n):
        solver.add_clause([x(i,j) for j in range(n)])

    # Ensure all nodes are in distinct slots
    for i in range(n):
        for j in range(n):
            for l in range(n):
                if j == l:
                    continue
                solver.add_clause([-x(i,j), -x(i,l)])

    # Ensure non-adjacent nodes are not adjacent in path
    for j in range(n):
        for l in range(n):
            if not G.has_edge(list(G)[j], list(G)[l]):
                for i in range(n-1):
                    solver.add_clause([-x(i,j), -x(i+1,l)])
    
    # Ensure first and last are connected by an edge
    for j in range(n):
        for l in range(n):
            if not G.has_edge(list(G)[j], list(G)[l]):
                solver.add_clause([-x(0,j),-x(n-1,l)])

    return solver

def hamCyc(G: nx.Graph) -> list[int]:
    solver = hamCycSolver(G)

    n = len(G)
    index = lambda i,j: i * n + j

    if not solver.solve():
        return None

    model = solver.get_model()

    return [list(G)[j] for i in range(n) for j in range(n) if model[index(i,j)] > 0]

def hasHamCyc(G: nx.Graph) -> bool:
    solver = hamCycSolver(G)
    return solver.solve()