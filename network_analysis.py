import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path

def load_pivot():
    df = pd.read_csv("aggregated_real_estate_volume_by_year.csv")
    pivot = df.pivot(index='IME_KO', columns='YEAR', values='TOTAL_VOLUME')
    return pivot

def load_graph():
    adj = pd.read_csv("ljubljana_cadastral_neighbors.csv")
    edges = adj[['IME_KO_A','IME_KO_B']].drop_duplicates().values.tolist()
    nodes = pd.read_csv("aggregated_real_estate_volume_by_year.csv")['IME_KO'].unique().tolist()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def compute_changes(pivot):
    years = sorted(pivot.columns)
    changes = pd.DataFrame(index=pivot.index, columns=years)
    for i, y in enumerate(years):
        if i == 0:
            changes[y] = np.nan
        else:
            changes[y] = pivot[y] - pivot[years[i-1]]
    return changes

def morans_i(values, G):
    x = values.values
    n = len(x)
    w = nx.to_numpy_array(G, nodelist=values.index)
    w_sum = w.sum()
    x_mean = np.nanmean(x)
    diff = x - x_mean
    num = 0
    for i in range(n):
        for j in range(n):
            if not np.isnan(diff[i]) and not np.isnan(diff[j]):
                num += w[i,j] * diff[i] * diff[j]
    den = np.nansum(diff**2)
    return (n / w_sum) * num / den

def diffusion_errors(pivot, G):
    A = nx.to_numpy_array(G, nodelist=pivot.index)
    deg = A.sum(axis=1)
    inv_deg = np.where(deg > 0, 1 / deg, 0)
    D = np.diag(inv_deg)
    M = D @ A
    years = sorted(pivot.columns)
    errors = {}
    for i in range(len(years) - 1):
        y = years[i]
        y_next = years[i+1]
        prev = pivot[y].fillna(0).values
        pred = M @ prev
        actual = pivot[y_next].fillna(0).values
        errors[y_next] = np.mean((actual - pred) ** 2)
    return errors

def leader_counts(changes, G):
    lead = {n: 0 for n in G.nodes}
    for y in sorted(changes.columns)[1:]:
        for u, v in G.edges:
            cu = changes.at[u, y]
            cv = changes.at[v, y]
            if pd.notna(cu) and pd.notna(cv):
                if cu > cv:
                    lead[u] += 1
                elif cv > cu:
                    lead[v] += 1
    return lead

def first_movers(changes):
    count = {n: 0 for n in changes.index}
    for y in sorted(changes.columns)[1:]:
        vals = changes[y].dropna()
        if not vals.empty:
            mover = vals.idxmax()
            count[mover] += 1
    return count

def main():
    pivot = load_pivot()
    G = load_graph()
    changes = compute_changes(pivot)
    for y in sorted(changes.columns)[1:]:
        I = morans_i(changes[y].fillna(0), G)
        print(y, I)
    errors = diffusion_errors(pivot, G)
    for y, e in errors.items():
        print("error", y, e)
    lc = leader_counts(changes, G)
    for n, c in sorted(lc.items(), key=lambda x: x[1], reverse=True):
        print("lead", n, c)
    fm = first_movers(changes)
    for n, c in sorted(fm.items(), key=lambda x: x[1], reverse=True):
        print("first", n, c)

if __name__ == "__main__":
    main()
