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
    w = weights.remap_ids(w, list(changes.index))
    results = {}
    for year in changes.columns[1:]:
        y = changes[year].dropna()
        common = y.index.intersection(w.id_order)
        y_sub = y.loc[common].values
        w_sub = weights.w_subset(w, common.tolist())
        moran = esda.Moran(y_sub, w_sub)
        results[year] = moran
    return results

def local_morans_i(G, changes, significance_level=0.05):
    w = weights.W.from_networkx(G)
    w.transform = 'r'
    w = weights.remap_ids(w, list(changes.index))
    results = {}
    for year in changes.columns[1:]:
        y = changes[year].dropna()
        common = y.index.intersection(w.id_order)
        y_sub = y.loc[common].values
        w_sub = weights.w_subset(w, common.tolist())
        local = esda.Moran_Local(y_sub, w_sub)
        results[year] = (local, common)
    return results

def plot_global_morans_i(global_morans, alpha=0.05):
    years = list(global_morans.keys())
    I_vals = [global_morans[y].I for y in years]
    p_vals = [global_morans[y].p_sim for y in years]
    plt.figure(figsize=(10, 6))
    plt.plot(years, I_vals, 'o-', linewidth=2, label="Global Moran's I")
    for x, I, p in zip(years, I_vals, p_vals):
        color = 'red' if p < alpha else 'black'
        plt.plot(x, I, 'o', color=color)
        plt.text(x, I + 0.01, f"p={p:.3f}", ha='center', fontsize=9)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel("Moran's I")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_morans_i.png", dpi=300)
    plt.close()

def plot_local_morans_i(local_morans, year, G, changes, alpha=0.05):
    local, nodes = local_morans[year]
    sig = local.p_sim < alpha
    labels = local.q
    values = changes.loc[nodes, year]
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42, k=0.1)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', width=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray', alpha=0.5)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodes,
        node_color=[cm.viridis(colors.Normalize(vmin=values.min(), vmax=values.max())(v)) for v in values],
        node_size=100,
        alpha=0.9
    )
    cluster_colors = {1:'red', 2:'lightblue', 3:'blue', 4:'orange'}
    cluster_names = {1:'High-High', 2:'Low-High', 3:'Low-Low', 4:'High-Low'}
    for q in cluster_colors:
        cnodes = [nodes[i] for i in range(len(nodes)) if labels[i]==q and sig[i]]
        if cnodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=cnodes,
                node_size=120,
                node_color='none',
                edgecolors=cluster_colors[q],
                linewidths=2,
                label=cluster_names[q]
            )
    nx.draw_networkx_labels(G, pos, font_size=7, font_color='black', alpha=0.7)
    plt.legend(scatterpoints=1)
    fig = plt.gcf()
    ax = plt.gca()
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=colors.Normalize(vmin=values.min(), vmax=values.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Change')
    plt.axis('off')
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
    plot_global_morans_i(gmi, alpha=0.05)
    last_year = changes.columns[-1]
    lmi = local_morans_i(G, changes, significance_level=0.05)
    plot_local_morans_i(lmi, last_year, G, changes, alpha=0.05)
    errors = diffusion_errors(pivot, G)
    plot_diffusion_errors(errors)
    lc = leader_counts(changes, G)
    plot_leader_counts(lc)
    fm = first_movers(changes)
    plot_first_movers(fm)


if __name__ == "__main__":
    main()