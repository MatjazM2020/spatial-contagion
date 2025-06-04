import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

plt.style.use('ggplot')
sns.set_palette('viridis')
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

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

def morans_i(values, G, permutations=999, seed=42):
    rng = np.random.default_rng(seed)
    x = values.values
    w = nx.to_numpy_array(G, nodelist=values.index)
    x_mean = np.nanmean(x)
    diff = x - x_mean
    
    valid = ~np.isnan(diff)
    diff_valid = diff[valid]
    w_valid = w[np.ix_(valid, valid)]
    
    num = np.sum(w_valid * np.outer(diff_valid, diff_valid))
    den = np.sum(diff_valid ** 2)
    observed_I = (len(diff_valid) / w_valid.sum()) * (num / den)
    
    permuted_Is = []
    for _ in range(permutations):
        permuted = rng.permutation(diff_valid)
        num_perm = np.sum(w_valid * np.outer(permuted, permuted))
        I_perm = (len(diff_valid) / w_valid.sum()) * (num_perm / den)
        permuted_Is.append(I_perm)

    permuted_Is = np.array(permuted_Is)
    p_value = np.mean(permuted_Is >= observed_I)
    
    return observed_I, p_value

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

def granger_spillover(pivot, G, lag=1):
    results = []
    for u, v in G.edges():
        if u not in pivot.index or v not in pivot.index:
            continue
        df_edge = pivot.loc[[u, v]].T.dropna()
        if len(df_edge) < lag + 5:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                gc_res = grangercausalitytests(df_edge[[v, u]], maxlag=lag, verbose=False)
                p_forward = gc_res[lag][0]['ssr_ftest'][1]
            except:
                p_forward = 1.0
            try:
                gc_res = grangercausalitytests(df_edge[[u, v]], maxlag=lag, verbose=False)
                p_reverse = gc_res[lag][0]['ssr_ftest'][1]
            except:
                p_reverse = 1.0

        if p_forward < 0.05:
            results.append((u, v, p_forward, "->"))
        if p_reverse < 0.05:
            results.append((v, u, p_reverse, "->"))
    return results

def plot_morans_i(results, filename=None, alpha=0.05):
    plt.figure(figsize=(10, 6))
    
    years = list(results.keys())
    morans_i = [v[0] for v in results.values()]
    p_values = [v[1] for v in results.values()]
    
    significant = [p < alpha for p in p_values]
    plt.plot(years, morans_i, 'o-', linewidth=2, label="Moran's I")
    
    for i, (x, y, sig, p) in enumerate(zip(years, morans_i, significant, p_values)):
        color = 'red' if sig else 'black'
        plt.plot(x, y, 'o', color=color)
        plt.text(x, y + 0.01, f"p={p:.3f}", ha='center', fontsize=9)
    
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("Moran's I Spatial Autocorrelation (with p-values)")
    plt.xlabel('Year')
    plt.ylabel("Moran's I")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_diffusion_errors(errors, filename=None):
    plt.figure(figsize=(10, 6))
    years = list(errors.keys())
    values = list(errors.values())
    plt.plot(years, values, 's-', linewidth=2)
    plt.title("Diffusion Model Prediction Error")
    plt.xlabel('Year')
    plt.ylabel("Mean Squared Error (MSE)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_leader_counts(counts, filename=None):
    top20 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
    communities, values = zip(*top20)
    plt.figure(figsize=(12, 8))
    plt.bar(communities, values)
    plt.title("Top 20 Communities by Neighborhood Leadership Count")
    plt.xlabel('Community')
    plt.ylabel('Leadership Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_first_movers(counts, filename=None):
    top20 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
    communities, values = zip(*top20)
    plt.figure(figsize=(12, 8))
    plt.bar(communities, values)
    plt.title("Top 20 First Mover Communities")
    plt.xlabel('Community')
    plt.ylabel('First Mover Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_granger_spillover(results, filename=None):
    if not results:
        print("No significant Granger causality relationships found.")
        return
    D = nx.DiGraph()
    for u, v, p, _ in results:
        D.add_edge(u, v, weight=-np.log10(p))

    out_degrees = dict(D.out_degree())
    sizes = [300 + out_degrees[n] * 1000 for n in D.nodes()]

    pos = nx.spring_layout(D, k=0.5, seed=42)
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(D, pos, node_size=sizes, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(D, pos, arrowstyle='->', arrowsize=10, edge_color='gray', width=1)
    nx.draw_networkx_labels(D, pos, font_size=8)

    plt.title("Granger Causality Network")
    plt.axis('off')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def main():
    pivot = load_pivot()
    G = load_graph()
    changes = compute_changes(pivot)
    
    morans_results = {}
    for y in sorted(changes.columns)[1:]:
        I, p = morans_i(changes[y].fillna(0), G)
        morans_results[y] = (I, p)
        print(f"{y} Moran's I: {I:.4f}, p-value: {p:.4f}")
    plot_morans_i(morans_results, OUTPUT_DIR / "morans_i.png")
    
    errors = diffusion_errors(pivot, G)
    for y, e in errors.items():
        print(f"Diffusion error ({y}): {e:.2f}")
    plot_diffusion_errors(errors, OUTPUT_DIR / "diffusion_errors.png")
    
    lc = leader_counts(changes, G)
    for n, c in sorted(lc.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Leader: {n} ({c})")
    plot_leader_counts(lc, OUTPUT_DIR / "leader_counts.png")
    
    fm = first_movers(changes)
    for n, c in sorted(fm.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"First mover: {n} ({c})")
    plot_first_movers(fm, OUTPUT_DIR / "first_movers.png")
    
    lag=1
    gc_results = granger_spillover(pivot, G, lag=lag)
    for u, v, p, direction in gc_results:
        print(f"Granger: {u} {direction} {v} (p={p:.4f})")
    plot_granger_spillover(gc_results, OUTPUT_DIR / f"granger_spillover_lag_{lag}.png")

if __name__ == "__main__":
    main()
