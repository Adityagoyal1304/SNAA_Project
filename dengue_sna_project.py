import os
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from networkx.algorithms.community import greedy_modularity_communities

# ================================================================
# 1. LOAD DATASET
# ================================================================
data_dir = "./data"

# Download from Kaggle if data doesn't exist
if not os.path.exists(data_dir) or not any(f.endswith('.csv') for f in os.listdir(data_dir)):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "fazlyrabbi/dengue-incidents-weather-of-bangladesh",
            path=data_dir, unzip=True
        )
        print("Dataset downloaded from Kaggle.")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        exit(1)

file = [f for f in os.listdir(data_dir) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(data_dir, file))
print(f"Loaded: {file} ({len(df)} records)")

# Normalize column names
df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={
    "year": "year", "month": "month", "dengue": "cases",
    "min": "temp_min", "max": "temp_max",
    "humidity": "humidity", "rainfall": "rainfall"
})

# Create datetime and normalized cases
df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2))
df["case_norm"] = df["cases"] / df["cases"].max()
df["period"] = df["date"].dt.strftime("%Y-%m")
df["temp_avg"] = (df["temp_min"] + df["temp_max"]) / 2

print(f"Time range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total dengue cases: {df['cases'].sum():,}")
print(f"Peak month: {df.loc[df['cases'].idxmax(), 'period']} ({df['cases'].max():,} cases)")

# ================================================================
# 2. BUILD DIRECTED WEIGHTED GRAPH
# ================================================================
G = nx.DiGraph()

for _, row in df.iterrows():
    G.add_node(row["date"], period=row["period"], cases=int(row["cases"]),
               temp_avg=row["temp_avg"], rainfall=row["rainfall"],
               humidity=row["humidity"], year=int(row["year"]),
               month=int(row["month"]))

def sim(a, b):
    return 1 / (1 + abs(a - b))

def compute_weight(r1, r2):
    temp_sim = sim(r1["temp_avg"], r2["temp_avg"])
    rain_sim = sim(r1["rainfall"], r2["rainfall"])
    hum_sim  = sim(r1["humidity"], r2["humidity"])
    climate_sim = (temp_sim + rain_sim + hum_sim) / 3
    # Weight combines climate similarity with source case burden
    return climate_sim * r1["case_norm"]

rows = df.to_dict("records")
threshold = 0.05  # Lower threshold for denser network

for i in range(len(rows)):
    for j in range(len(rows)):
        if i == j:
            continue
        w = compute_weight(rows[i], rows[j])
        if w > threshold:
            G.add_edge(rows[i]["date"], rows[j]["date"], weight=w)

print(f"\n{'='*50}")
print("NETWORK CONSTRUCTION")
print(f"{'='*50}")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# ================================================================
# 3. COMPREHENSIVE SNA METRICS
# ================================================================
print(f"\n{'='*50}")
print("SNA METRICS")
print(f"{'='*50}")

# Node-level centrality
pagerank    = nx.pagerank(G, weight="weight")
in_deg_c    = nx.in_degree_centrality(G)
out_deg_c   = nx.out_degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
closeness   = nx.closeness_centrality(G)
clustering  = nx.clustering(G)

# HITS
try:
    hubs, authorities = nx.hits(G, max_iter=1000)
except:
    hubs = {n: 0 for n in G.nodes()}
    authorities = {n: 0 for n in G.nodes()}

# Network-level stats
density = nx.density(G)
print(f"Network Density: {density:.4f}")

sccs = list(nx.strongly_connected_components(G))
print(f"Strongly Connected Components: {len(sccs)}")
largest_scc = max(sccs, key=len)
print(f"Largest SCC size: {len(largest_scc)}")

wccs = list(nx.weakly_connected_components(G))
print(f"Weakly Connected Components: {len(wccs)}")

scc_sub = G.subgraph(largest_scc)
if nx.is_strongly_connected(scc_sub):
    diameter = nx.diameter(scc_sub)
    avg_path = nx.average_shortest_path_length(scc_sub)
    print(f"Diameter (largest SCC): {diameter}")
    print(f"Avg Shortest Path (largest SCC): {avg_path:.4f}")
else:
    diameter = "N/A"
    avg_path = "N/A"
    print(f"Diameter: {diameter}")

avg_clust = nx.average_clustering(G)
print(f"Average Clustering: {avg_clust:.4f}")

reciprocity = nx.reciprocity(G)
print(f"Reciprocity: {reciprocity:.4f}")

# Community detection
communities = list(greedy_modularity_communities(G.to_undirected()))
modularity = nx.community.modularity(G.to_undirected(), communities)
print(f"Communities: {len(communities)}")
print(f"Modularity: {modularity:.4f}")

community_map = {}
for idx, comm in enumerate(communities):
    for node in comm:
        community_map[node] = idx

# Top PageRank
top_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"\nTop 10 Outbreak Periods (PageRank):")
for d, v in top_pr:
    print(f"  {d.strftime('%Y-%m')}  PR={v:.4f}  Cases={G.nodes[d]['cases']}")

top_bw = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"\nTop 5 Bridge Periods (Betweenness):")
for d, v in top_bw:
    print(f"  {d.strftime('%Y-%m')}  BW={v:.4f}")

# Print top 4 community compositions
for i, comm in enumerate(communities[:4]):
    periods = sorted([n.strftime('%Y-%m') for n in comm])
    print(f"\nCommunity {i+1} ({len(comm)} periods): {', '.join(periods[:10])}...")
if len(communities) > 4:
    print(f"\n... and {len(communities)-4} more small communities")

# ================================================================
# 4. EXPORT CSV
# ================================================================
metrics_data = []
for node in G.nodes():
    metrics_data.append({
        "Period": node.strftime("%Y-%m"),
        "Year": G.nodes[node]["year"],
        "Month": G.nodes[node]["month"],
        "Cases": G.nodes[node]["cases"],
        "In_Degree_C": round(in_deg_c[node], 4),
        "Out_Degree_C": round(out_deg_c[node], 4),
        "Betweenness": round(betweenness[node], 4),
        "Closeness": round(closeness[node], 4),
        "PageRank": round(pagerank[node], 4),
        "Clustering": round(clustering[node], 4),
        "Hub_Score": round(hubs[node], 4),
        "Authority_Score": round(authorities[node], 4),
        "Community": community_map.get(node, -1)
    })

metrics_df = pd.DataFrame(metrics_data).sort_values("Period")
metrics_df.to_csv("sna_metrics.csv", index=False)
print(f"\nExported: sna_metrics.csv ({len(metrics_df)} rows)")

# ================================================================
# 5. VISUALIZATIONS
# ================================================================
DARK_BG = '#1a1a2e'
DARK_FG = '#e0e0e0'
COMM_COLORS = ['#e94560', '#537FE7', '#16c79a', '#f5a623', '#8b5cf6', '#ec4899']

pos = nx.spring_layout(G, seed=42, k=2.5/np.sqrt(G.number_of_nodes()))

# --- Fig 1: Transmission Network ---
fig1, ax1 = plt.subplots(figsize=(14, 10), facecolor=DARK_BG)
ax1.set_facecolor(DARK_BG)

node_colors = [COMM_COLORS[community_map.get(n, 0) % len(COMM_COLORS)] for n in G.nodes()]
max_pr = max(pagerank.values())
node_sizes = [2500 * (pagerank[n] / max_pr) + 30 for n in G.nodes()]

nx.draw_networkx_edges(G, pos, alpha=0.06, edge_color='#ffffff', width=0.3, arrows=False, ax=ax1)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       alpha=0.85, ax=ax1, edgecolors='white', linewidths=0.5)

labels = {d: d.strftime("%Y-%m") for d, _ in top_pr[:8]}
nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color='white', font_weight='bold', ax=ax1)

ax1.set_title("Dengue Temporal Transmission Network – Bangladesh\n"
              "Node size ∝ PageRank | Color = Community | Edge weight ∝ Climate Similarity × Case Burden",
              color=DARK_FG, fontsize=13, fontweight='bold', pad=15)

from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=COMM_COLORS[i], label=f'Community {i+1}')
                  for i in range(min(len(communities), 4))]
ax1.legend(handles=legend_patches, loc='lower left', fontsize=9,
           facecolor=DARK_BG, edgecolor='#444', labelcolor=DARK_FG)
ax1.axis('off')
fig1.tight_layout()
fig1.savefig("fig1_transmission_network.png", dpi=200, facecolor=DARK_BG, bbox_inches='tight')
plt.close(fig1)
print("Saved: fig1_transmission_network.png")

# --- Fig 2: Centrality Metrics Bar Charts ---
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10), facecolor=DARK_BG)
chart_data = [
    ("PageRank (Outbreak Influence)", pagerank, '#e94560'),
    ("Betweenness (Bridge Periods)", betweenness, '#537FE7'),
    ("Closeness (Reach Speed)", closeness, '#16c79a'),
    ("Clustering Coefficient", clustering, '#f5a623')
]
for ax, (title, metric, color) in zip(axes2.flat, chart_data):
    ax.set_facecolor(DARK_BG)
    top20 = sorted(metric.items(), key=lambda x: x[1], reverse=True)[:20]
    lbls = [d.strftime("%Y-%m") for d, _ in top20]
    vals = [v for _, v in top20]
    ax.barh(range(len(lbls)), vals, color=color, alpha=0.85)
    ax.set_yticks(range(len(lbls)))
    ax.set_yticklabels(lbls, fontsize=7, color=DARK_FG)
    ax.set_xlabel("Score", color=DARK_FG, fontsize=9)
    ax.set_title(title, color=DARK_FG, fontsize=11, fontweight='bold')
    ax.tick_params(colors=DARK_FG, labelsize=8)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_color('#333')

fig2.suptitle("Centrality Metrics – Dengue Temporal Network",
              color=DARK_FG, fontsize=14, fontweight='bold')
fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig("fig2_centrality_metrics.png", dpi=200, facecolor=DARK_BG, bbox_inches='tight')
plt.close(fig2)
print("Saved: fig2_centrality_metrics.png")

# --- Fig 3: Seasonal Trends & Heatmap ---
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [1.2, 1]})

ax3a.fill_between(df["date"], df["cases"], alpha=0.3, color='#e94560')
ax3a.plot(df["date"], df["cases"], color='#e94560', linewidth=1.5)
ax3a.set_title("Monthly Dengue Cases – Bangladesh (National Level)", fontsize=13, fontweight='bold')
ax3a.set_ylabel("Confirmed Cases")
ax3a.grid(alpha=0.3)
ax3a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3a.xaxis.set_major_locator(mdates.YearLocator())

month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot = df.pivot_table(values="cases", index="year", columns="month", aggfunc="sum").fillna(0)
pivot.columns = [month_names[int(c)-1] for c in pivot.columns]
sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f', linewidths=0.5, ax=ax3b,
            cbar_kws={'label': 'Cases'}, annot_kws={'size': 8})
ax3b.set_title("Cases Heatmap (Year × Month)", fontsize=13, fontweight='bold')
ax3b.set_ylabel("Year")
ax3b.set_xlabel("Month")

fig3.tight_layout()
fig3.savefig("fig3_seasonal_trends.png", dpi=200, bbox_inches='tight')
plt.close(fig3)
print("Saved: fig3_seasonal_trends.png")

# --- Fig 4: Degree Distribution ---
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)

in_degs = [d for _, d in G.in_degree()]
out_degs = [d for _, d in G.out_degree()]

for ax, degs, title, color in [(ax4a, in_degs, "In-Degree Distribution", '#e94560'),
                                (ax4b, out_degs, "Out-Degree Distribution", '#537FE7')]:
    ax.set_facecolor(DARK_BG)
    ax.hist(degs, bins=range(min(degs), max(degs)+2), color=color, alpha=0.85,
            edgecolor=DARK_BG, rwidth=0.85)
    ax.set_title(title, color=DARK_FG, fontsize=12, fontweight='bold')
    ax.set_xlabel("Degree", color=DARK_FG)
    ax.set_ylabel("Frequency", color=DARK_FG)
    ax.tick_params(colors=DARK_FG)
    for spine in ax.spines.values():
        spine.set_color('#333')

fig4.suptitle("Degree Distribution – Dengue Temporal Network",
              color=DARK_FG, fontsize=14, fontweight='bold')
fig4.tight_layout(rect=[0, 0, 1, 0.93])
fig4.savefig("fig4_degree_distribution.png", dpi=200, facecolor=DARK_BG, bbox_inches='tight')
plt.close(fig4)
print("Saved: fig4_degree_distribution.png")

# --- network.png (simple version) ---
fig_n, ax_n = plt.subplots(figsize=(10, 6))
sizes = [5000*pagerank[n] for n in G.nodes()]
nx.draw(G, pos, node_size=sizes, with_labels=False, ax=ax_n,
        node_color=node_colors, edge_color='gray', alpha=0.7, width=0.2)
ax_n.set_title("Dengue Spread (Time-based Network)")
fig_n.savefig("network.png", dpi=150)
plt.close(fig_n)
print("Saved: network.png")

print("\nDONE - ALL OUTPUTS GENERATED SUCCESSFULLY")