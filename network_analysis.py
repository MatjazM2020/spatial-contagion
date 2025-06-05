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


def spatial_lag_model(pivot, G):
    """
    Fit Spatial Lag Models for each year transition to analyze spatial dependencies
    """
    # Convert NetworkX graph to libpysal weights
    adj_matrix = nx.to_numpy_array(G, nodelist=pivot.index)
    
    # Create neighbor dictionary for libpysal
    neighbors = {}
    for i in range(len(pivot.index)):
        neighbors[i] = list(np.where(adj_matrix[i] > 0)[0])
    
    # Create weights object
    w = W(neighbors)
    w.transform = 'r'  # Row standardization
    
    years = sorted(pivot.columns)
    slm_results = {}
    
    for i in range(len(years) - 1):
        y_current = years[i]
        y_next = years[i + 1]
        
        # Prepare data - remove NaN values
        data_current = pivot[y_current].fillna(0)
        data_next = pivot[y_next].fillna(0)
        
        # Create aligned arrays - only use observations without NaN in either year
        valid_mask = ~(pd.isna(pivot[y_current]) | pd.isna(pivot[y_next]))
        
        if valid_mask.sum() < 10:  # Need minimum observations
            print(f"Insufficient valid observations for {y_current}-{y_next}: {valid_mask.sum()}")
            continue
            
        # Get valid data
        valid_indices = np.where(valid_mask)[0]
        y_dep = data_next.iloc[valid_indices].values.reshape(-1, 1)
        x_indep = data_current.iloc[valid_indices].values.reshape(-1, 1)
        
        # Create filtered weights matrix for valid observations only
        w_filtered_neighbors = {}
        for new_i, orig_i in enumerate(valid_indices):
            # Find neighbors of original observation that are also in valid set
            orig_neighbors = neighbors[orig_i]
            valid_neighbors = []
            for neighbor in orig_neighbors:
                if neighbor in valid_indices:
                    # Map to new index in filtered dataset
                    new_neighbor_idx = np.where(valid_indices == neighbor)[0]
                    if len(new_neighbor_idx) > 0:
                        valid_neighbors.append(new_neighbor_idx[0])
            w_filtered_neighbors[new_i] = valid_neighbors
        
        # Create filtered weights object
        if len(w_filtered_neighbors) > 0:
            w_filtered = W(w_filtered_neighbors)
            w_filtered.transform = 'r'
        else:
            print(f"No valid neighbor connections for {y_current}-{y_next}")
            continue
        
        try:
            # Fit Spatial Lag Model
            slm = ML_Lag(y_dep, x_indep, w_filtered, name_y=f'Volume_{y_next}', 
                         name_x=[f'Volume_{y_current}'])
            
            slm_results[f'{y_current}-{y_next}'] = {
                'rho': slm.rho,  # spatial lag coefficient
                'beta': slm.betas[1][0],  # coefficient for previous year
                'pseudo_r2': slm.pr2,
                'log_likelihood': slm.logll,
                'aic': slm.aic,
                'n_obs': len(y_dep)
            }
            
        except Exception as e:
            print(f"SLM fitting failed for {y_current}-{y_next}: {e}")
            continue
    
    return slm_results


def plot_slm_results(slm_results, filename=None):
    """
    Visualize Spatial Lag Model results
    """
    if not slm_results:
        print("No SLM results to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    transitions = list(slm_results.keys())
    rhos = [slm_results[t]['rho'] for t in transitions]
    betas = [slm_results[t]['beta'] for t in transitions]
    pseudo_r2s = [slm_results[t]['pseudo_r2'] for t in transitions]
    aics = [slm_results[t]['aic'] for t in transitions]
    
    # Plot 1: Spatial lag coefficients (rho)
    ax1.bar(range(len(transitions)), rhos, color='skyblue', alpha=0.7)
    ax1.set_title('Spatial Lag Coefficients (ρ)')
    ax1.set_ylabel('ρ (Spatial Dependence)')
    ax1.set_xticks(range(len(transitions)))
    ax1.set_xticklabels(transitions, rotation=45, ha='right')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Autoregressive coefficients (beta)
    ax2.bar(range(len(transitions)), betas, color='lightcoral', alpha=0.7)
    ax2.set_title('Autoregressive Coefficients (β)')
    ax2.set_ylabel('β (Temporal Dependence)')
    ax2.set_xticks(range(len(transitions)))
    ax2.set_xticklabels(transitions, rotation=45, ha='right')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model fit (Pseudo R²)
    ax3.plot(range(len(transitions)), pseudo_r2s, 'o-', color='green', linewidth=2)
    ax3.set_title('Model Fit (Pseudo R²)')
    ax3.set_ylabel('Pseudo R²')
    ax3.set_xticks(range(len(transitions)))
    ax3.set_xticklabels(transitions, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(pseudo_r2s) * 1.1 if pseudo_r2s else 1)
    
    # Plot 4: Model comparison (AIC)
    ax4.plot(range(len(transitions)), aics, 's-', color='orange', linewidth=2)
    ax4.set_title('Model Selection Criterion (AIC)')
    ax4.set_ylabel('AIC (lower is better)')
    ax4.set_xticks(range(len(transitions)))
    ax4.set_xticklabels(transitions, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_spatial_temporal_dynamics(slm_results, filename=None):
    """
    Create a combined plot showing spatial vs temporal effects over time
    """
    if not slm_results:
        print("No SLM results to plot")
        return
        
    transitions = list(slm_results.keys())
    rhos = [slm_results[t]['rho'] for t in transitions]
    betas = [slm_results[t]['beta'] for t in transitions]
    
    plt.figure(figsize=(12, 8))
    
    x = range(len(transitions))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], rhos, width, label='Spatial Effect (ρ)', 
            color='skyblue', alpha=0.7)
    plt.bar([i + width/2 for i in x], betas, width, label='Temporal Effect (β)', 
            color='lightcoral', alpha=0.7)
    
    plt.title('Spatial vs Temporal Effects in Real Estate Volume Changes')
    plt.xlabel('Year Transitions')
    plt.ylabel('Coefficient Values')
    plt.xticks(x, transitions, rotation=45, ha='right')
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # Add text annotations for significant coefficients
    for i, (rho, beta) in enumerate(zip(rhos, betas)):
        if abs(rho) > 0.1:  # Highlight significant spatial effects
            plt.annotate(f'{rho:.3f}', (i - width/2, rho), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        if abs(beta) > 0.1:  # Highlight significant temporal effects
            plt.annotate(f'{beta:.3f}', (i + width/2, beta), 
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


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
    plt.bar(communities, values, color='blue')
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


def main():
    pivot = load_pivot()
    G = load_graph()
    changes = compute_changes(pivot)

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