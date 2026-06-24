import sys
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')

import community as cl
import leidenalg
import igraph as ig
import networkx as nx
import collections

print('Versions:')
print('  python-louvain:', getattr(cl, '__version__', 'unknown'))
print('  leidenalg:     ', leidenalg.version)
print('  python-igraph: ', ig.__version__)

G = nx.karate_club_graph()
p_louv = cl.best_partition(G, random_state=42, resolution=1.0)
G_ig = ig.Graph.from_networkx(G)
part_leid = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=42)
p_leid = {v: part_leid.membership[i] for i, v in enumerate(G.nodes())}

def signature(p):
    sizes = sorted(collections.Counter(p.values()).values(), reverse=True)
    return tuple(sizes)

print()
print('Karate Louvain partition: K=' + str(len(set(p_louv.values()))) + ', signature=' + str(signature(p_louv)))
print('Karate Leiden  partition: K=' + str(len(set(p_leid.values()))) + ', signature=' + str(signature(p_leid)))
print('Identical partition?', p_louv == p_leid)
