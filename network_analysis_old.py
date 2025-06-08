import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from libpysal.weights import W
from spreg import ML_Lag

plt.style.use('ggplot')
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

def plot_morans_i(results, filename=None, alpha=0.05):
    plt.figure(figsize=(10, 6))
    years = list(results.keys())
    morans_i = [v[0] for v in results.values()]
    p_values = [v[1] for v in results.values()]
    significant = [p < alpha for p in p_values]
    plt.plot(years, morans_i, 'o-', linewidth=2, label="Moran's I")
    for x, y, sig, p in zip(years, morans_i, significant, p_values):
        color = 'red' if sig else 'black'
        plt.plot(x, y, 'o', color=color)
        plt.text(x, y + 0.01, f"p={p:.3f}", ha='center', fontsize=9)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel("Moran's I")
    plt.xticks([2007 + a + 1 for a in range(2025 - 2007)], rotation=45)
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
    plt.xlabel('Year')
    plt.ylabel("Mean Squared Error (MSE)")
    plt.xticks([2007 + a + 1 for a in range(2025 - 2007)], rotation=45)
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
    #plt.title("Top 20 Communities by Neighborhood Leadership Count")
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
    #plt.title("Top 20 First Mover Communities")
    plt.xlabel('Community')
    plt.ylabel('First Mover Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_cadastral_network(G):
    degree_dict = dict(G.degree())
    scale_factor = 50
    
    node_sizes = [degree_dict[node] * scale_factor for node in G.nodes()] 
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42, k=0.1)
    nx.draw_networkx_edges(G, pos, width=0.5, edge_color='gray', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', edgecolors='black', linewidths=0.2, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=6, font_color='black', alpha=0.7)
    
    #plt.title("Cadastral Adjacency Network (Node size ∝ Degree)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "cadastral_network.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    pivot = load_pivot()
    G = load_graph()
    isolated = list(nx.isolates(G))
    if isolated:
        G.remove_nodes_from(isolated)
        pivot = pivot.drop(index=isolated)
    changes = compute_changes(pivot)

    plot_cadastral_network(G)

    # Moran's I
    morans_results = {}
    for y in sorted(changes.columns)[1:]:
        I, p = morans_i(changes[y].fillna(0), G)
        morans_results[y] = (I, p)
        print(f"{y} Moran's I: {I:.4f}, p-value: {p:.4f}")
    plot_morans_i(morans_results, OUTPUT_DIR / "morans_i.png")

    # Diffusion errors
    errors = diffusion_errors(pivot, G)
    for y, e in errors.items():
        print(f"Diffusion error ({y}): {e:.2f}")
    plot_diffusion_errors(errors, OUTPUT_DIR / "diffusion_errors.png")

    # Leader counts
    lc = leader_counts(changes, G)
    for n, c in sorted(lc.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Leader: {n} ({c})")
    plot_leader_counts(lc, OUTPUT_DIR / "leader_counts.png")

    # First movers
    fm = first_movers(changes)
    for n, c in sorted(fm.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"First mover: {n} ({c})")
    plot_first_movers(fm, OUTPUT_DIR / "first_movers.png")


if __name__ == "__main__":
    main()