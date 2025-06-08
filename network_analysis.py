import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import esda
from libpysal import weights

plt.style.use('ggplot')
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_pivot():
    df = pd.read_csv("aggregated_real_estate_volume_by_year.csv")
    pivot = df.pivot(index='IME_KO', columns='YEAR', values='TOTAL_VOLUME')
    return pivot

def load_graph(pivot):
    adj = pd.read_csv("ljubljana_cadastral_neighbors.csv")
    edges = adj[['IME_KO_A','IME_KO_B']].drop_duplicates().values.tolist()
    nodes = pivot.index.tolist()
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

def diffusion_errors(pivot, G):
    A = nx.to_numpy_array(G, nodelist=pivot.index)
    deg = A.sum(axis=1)
    inv_deg = np.where(deg > 0, 1.0 / deg, 0.0)
    D = np.diag(inv_deg)
    M = D @ A
    years = sorted(pivot.columns)
    errors = {}
    for i in range(len(years) - 1):
        y, y_next = years[i], years[i+1]
        prev = pivot[y].fillna(0).values
        actual = pivot[y_next].fillna(0).values
        pred = M @ prev
        errors[y_next] = np.mean((actual - pred) ** 2)
    calc_dir = OUTPUT_DIR / "calculations"
    calc_dir.mkdir(exist_ok=True)
    pd.Series(errors).to_csv(calc_dir / "diffusion_errors.csv", header=["mse"])
    return errors

def leader_counts(changes, G):
    lead = {n: 0 for n in G.nodes()}
    for year in sorted(changes.columns)[1:]:
        for u, v in G.edges():
            cu = changes.at[u, year]
            cv = changes.at[v, year]
            if pd.notna(cu) and pd.notna(cv):
                if cu > cv:
                    lead[u] += 1
                elif cv > cu:
                    lead[v] += 1
    calc_dir = OUTPUT_DIR / "calculations"
    calc_dir.mkdir(exist_ok=True)
    pd.Series(lead).to_csv(calc_dir / "leader_counts.csv", header=["leader_count"])
    return lead

def first_movers(changes):
    movers = {n: 0 for n in changes.index}
    for year in sorted(changes.columns)[1:]:
        vals = changes[year].dropna()
        if not vals.empty:
            top = vals.idxmax()
            movers[top] += 1
    calc_dir = OUTPUT_DIR / "calculations"
    calc_dir.mkdir(exist_ok=True)
    pd.Series(movers).to_csv(calc_dir / "first_movers.csv", header=["first_mover_count"])
    return movers

def plot_cadastral_network(G, pivot):
    degree_dict = dict(G.degree())
    volume_sum = pivot.sum(axis=1)
    scale = 50
    node_sizes = [degree_dict[n] * scale for n in G.nodes()]
    norm = colors.Normalize(vmin=volume_sum.min(), vmax=volume_sum.max())
    cmap = cm.viridis
    node_colors = [cmap(norm(volume_sum[n])) for n in G.nodes()]
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42, k=0.1)
    nx.draw_networkx_edges(G, pos, width=0.5, edge_color='gray', alpha=0.5)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='black',
        linewidths=0.2,
        alpha=0.8
    )
    nx.draw_networkx_labels(G, pos, font_size=6, font_color='black', alpha=0.7)
    fig = plt.gcf()
    ax = plt.gca()
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Total Volume')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cadastral_network.png", dpi=300, bbox_inches='tight')
    plt.close()

def global_morans_i(G, changes):
    w = weights.W.from_networkx(G)
    w.transform = 'r'
    w = weights.remap_ids(w, changes.index)
    results = {}
    years = changes.columns[1:]
    for year in years:
        y = changes[year].dropna()
        common_index = y.index.intersection(w.id_order)
        y = y.loc[common_index]
        w_sub = weights.w_subset(w, common_index.tolist())
        moran = esda.Moran(y.values, w_sub)
        results[year] = moran

    return results

def local_morans_i(G, changes):
    w = weights.W.from_networkx(G)
    w.transform = 'r'
    w = weights.remap_ids(w, changes.index)
    results = {}
    years = changes.columns[1:]
    for year in years:
        y = changes[year].dropna()
        common_index = y.index.intersection(w.id_order)
        y = y.loc[common_index]
        w_sub = weights.w_subset(w, common_index.tolist())
        local_moran = esda.Moran_Local(y.values, w_sub)
        results[year] = results[year] = (local_moran, common_index)
    return results

def plot_global_morans_i(global_morans, significance_level=0.05):
    years = list(global_morans.keys())
    values = [global_morans[year].I for year in years]
    p_values = [global_morans[year].p_sim for year in years]
    plt.figure(figsize=(10, 6))
    plt.plot(years, values, marker='o', linestyle='-', label="Moran's I")
    for i, p in enumerate(p_values):
        if p < significance_level:
            plt.annotate('*', (years[i], values[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=14, color='blue')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Global Moran's I over Years")
    plt.xlabel("Year")
    plt.ylabel("Moran's I")
    plt.xticks(years)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_morans_i.png", dpi=300)
    plt.close()


def plot_local_morans_i(local_morans, year, G, changes, significance_level=0.05):
    local_moran, node_index = local_morans[year]
    sig = local_moran.p_sim < significance_level
    cluster_labels = local_moran.q
    pos = nx.spring_layout(G, seed=42, k=0.1)
    plt.figure(figsize=(12, 12))
    values = changes.loc[node_index, year]
    norm = colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.viridis
    node_colors = [cmap(norm(v)) for v in values]
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=50, alpha=0.5)
    nx.draw_networkx_edges(G, pos, width=0.5, edge_color='gray', alpha=0.3)
    nx.draw_networkx_nodes(G, pos, nodelist=node_index, node_color=node_colors, node_size=100, alpha=0.9)
    cluster_edge_colors = {1: 'red', 2: 'lightblue', 3: 'blue', 4: 'orange'}
    labels = {1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low'}
    for cluster_type in [1, 2, 3, 4]:
        cluster_nodes = [node_index[i] for i in range(len(node_index)) if cluster_labels[i] == cluster_type and sig[i]]
        if cluster_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, node_size=120, node_color='none', edgecolors=cluster_edge_colors[cluster_type], linewidths=2, alpha=1, label=labels[cluster_type])
    nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()}, font_size=7, font_color='black', alpha=0.7)
    plt.title(f"Local Moran's I Clusters for Year {year}")
    plt.legend(scatterpoints=1)
    plt.axis('off')
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax = plt.gca()
    plt.colorbar(sm, ax=ax, shrink=0.7, label='Change in volume')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"local_morans_i_{year}.png", dpi=300)
    plt.close()


def plot_diffusion_errors(errors):
    years, errs = zip(*sorted(errors.items()))
    plt.figure(figsize=(10, 6))
    plt.plot(years, errs, 's-', linewidth=2)
    plt.xlabel('Year')
    plt.xticks(years, rotation=45)
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "diffusion_errors.png", dpi=300)
    plt.close()

def plot_leader_counts(counts):
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
    comms, vals = zip(*top)
    plt.figure(figsize=(12, 8))
    plt.bar(comms, vals)
    plt.xlabel('Community')
    plt.ylabel('Leader Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "leader_counts.png", dpi=300)
    plt.close()

def plot_first_movers(counts):
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
    comms, vals = zip(*top)
    plt.figure(figsize=(12, 8))
    plt.bar(comms, vals)
    plt.xlabel('Community')
    plt.ylabel('First Mover Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "first_movers.png", dpi=300)
    plt.close()


def main():
    pivot = load_pivot()
    print("Total number of distinct cadastral units:", len(pivot.index))
    G = load_graph(pivot)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    pivot = pivot.drop(index=isolates)
    print("Number of connected cadastral units:", len(pivot.index))
    print("Number of nodes in final connected graph:", G.number_of_nodes())
    print("Number of edges in final connected graph:", G.number_of_edges())
    changes = compute_changes(pivot)
    plot_cadastral_network(G, pivot)
    gmi = global_morans_i(G, changes)
    plot_global_morans_i(gmi)
    last_year = changes.columns[-1]
    lmi = local_morans_i(G, changes)
    plot_local_morans_i(lmi, last_year, G, changes)
    errors = diffusion_errors(pivot, G)
    plot_diffusion_errors(errors)
    lc = leader_counts(changes, G)
    plot_leader_counts(lc)
    fm = first_movers(changes)
    plot_first_movers(fm)


if __name__ == "__main__":
    main()