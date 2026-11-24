"""Full-support integer min-cost fallback with a verified dual witness."""
from fractions import Fraction as F
from math import lcm
import networkx as nx

from .certified_reference import Band


def solve(a,b,c,bits=76):
    nr,nc=len(a),len(b)
    den=lcm(*(p.denominator for p in a+b))
    scale=1<<bits
    graph=nx.DiGraph()
    for i,p in enumerate(a):
        graph.add_node(i,demand=-int(den*p))
    for j,q in enumerate(b):
        graph.add_node(nr+j,demand=int(den*q))
    weights=[[cell.lo.numerator*scale//cell.lo.denominator for cell in row] for row in c]
    for i in range(nr):
        for j in range(nc):
            graph.add_edge(i,nr+j,weight=weights[i][j],capacity=den)
    objective,plan=nx.network_simplex(graph)
    # For fixed integer costs, find potentials from all-cell inequalities and
    # tightness on transported cells. Initial zero potentials act as a super-source.
    potentials=[0]*(nr+nc)
    for iteration in range(nr+nc):
        changed=False
        for i in range(nr):
            for j in range(nc):
                k=nr+j
                weight=weights[i][j]
                if potentials[i]>potentials[k]+weight:
                    potentials[i]=potentials[k]+weight
                    changed=True
                if plan[i][k] and potentials[k]>potentials[i]-weight:
                    potentials[k]=potentials[i]-weight
                    changed=True
        if not changed:
            break
    if any(potentials[i]-potentials[nr+j]>weights[i][j] for i in range(nr) for j in range(nc)):
        raise ArithmeticError("integer transportation dual infeasible")
    u=potentials[:nr]
    v=[-p for p in potentials[nr:]]
    dual=sum(p*z for p,z in zip(a,u))+sum(q*z for q,z in zip(b,v))
    if dual!=F(objective,den):
        raise ArithmeticError("integer transportation primal-dual gap")
    if (any(sum(plan[i].values())!=den*a[i] for i in range(nr)) or
        any(sum(plan[i][nr+j] for i in range(nr))!=den*b[j] for j in range(nc))):
        raise ArithmeticError("integer transportation marginal error")
    high=sum(F(plan[i][nr+j],den)*c[i][j].hi for i in range(nr) for j in range(nc))
    return Band(dual/scale,high)
