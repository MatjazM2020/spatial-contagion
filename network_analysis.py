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

def plot_cadastral_network(G, pivot):
    degree_dict = dict(G.degree())
    volume_sum = pivot.sum(axis=1)
    scale_factor = 50
    
    node_sizes = [degree_dict[node] * scale_factor for node in G.nodes()]
    
    norm = colors.Normalize(vmin=volume_sum.min(), vmax=volume_sum.max())
    cmap = cm.viridis
    
    node_colors = [cmap(norm(volume_sum[node])) if node in volume_sum else cmap(0) for node in G.nodes()]
    
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42, k=0.1)
    
    nx.draw_networkx_edges(G, pos, width=0.5, edge_color='gray', alpha=0.5)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                   node_color=node_colors,
                                   edgecolors='black', linewidths=0.2, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=6, font_color='black', alpha=0.7)
    
    plt.axis('off')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Volume')
    
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

def local_morans_i(G, changes, significance_level=0.05):
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
            plt.annotate('*', (years[i], values[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=14, color='red')

    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("Global Moran's I over Years")
    plt.xlabel("Year")
    plt.ylabel("Moran's I")
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

    nx.draw_networkx_nodes(G, pos,
                           nodelist=node_index,
                           node_color=node_colors,
                           node_size=100,
                           alpha=0.9)

    cluster_edge_colors = {1: 'red', 2: 'lightblue', 3: 'blue', 4: 'orange'}
    labels = {1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low'}

    for cluster_type in [1, 2, 3, 4]:
        cluster_nodes = [node_index[i] for i in range(len(node_index))
                         if cluster_labels[i] == cluster_type and sig[i]]
        if cluster_nodes:
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=cluster_nodes,
                                   node_size=120,
                                   node_color='none',
                                   edgecolors=cluster_edge_colors[cluster_type],
                                   linewidths=2,
                                   alpha=1,
                                   label=labels[cluster_type])

    nx.draw_networkx_labels(G, pos,
                            labels={node: node for node in G.nodes()},
                            font_size=7,
                            font_color='black',
                            alpha=0.7)

    plt.title(f"Local Moran's I Clusters for Year {year}\n(Significant at p<{significance_level})")
    plt.legend(scatterpoints=1)
    plt.axis('off')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax = plt.gca()
    plt.colorbar(sm, ax=ax, shrink=0.7, label='Change Value')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"local_morans_i_{year}.png", dpi=300)
    plt.close()


def main():
    pivot = load_pivot()
    G = load_graph(pivot)
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    pivot = pivot.drop(index=isolated)
    changes = compute_changes(pivot)

    plot_cadastral_network(G, pivot)    

    global_morans = global_morans_i(G, changes)
    plot_global_morans_i(global_morans, significance_level=0.05)

    local_morans = local_morans_i(G, changes)
    plot_local_morans_i(local_morans, 2025, G, changes, significance_level=0.05)

if __name__ == "__main__":
    main()