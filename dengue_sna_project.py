"""
=======================================================================
SOCIAL NETWORK ANALYSIS PROJECT
TITLE  : Dengue Disease Spread Network Analysis – Bangladesh
DOMAIN : Disease Spread / Epidemiological Network Analysis
DATASET: Dengue Incidents & Weather Data of Bangladesh (Kaggle)
         https://www.kaggle.com/datasets/fazlyrabbi/dengue-incidents-weather-of-bangladesh
=======================================================================

WHAT THIS PROJECT DOES
-----------------------
Models the spread of Dengue fever across Bangladeshi districts as a
weighted, directed network graph where:
  - NODES  = Districts (administrative regions)
  - EDGES  = Disease transmission pathways (weighted by case counts
             and geographic/climatic proximity)
  - WEIGHT = Strength of transmission link (normalized case burden)

KEY SNA METRICS COMPUTED
-------------------------
1. Degree Centrality        – Which districts drive the most connections?
2. Betweenness Centrality   – Which districts act as "bridges" in spread?
3. Closeness Centrality     – Which districts can reach others fastest?
4. PageRank                 – Recursive influence / super-spreader score
5. Clustering Coefficient   – Local cluster density
6. Community Detection      – Louvain-style epidemic clusters
7. Network Diameter         – Maximum hops across the spread network
8. Average Path Length      – Speed of transmission across network
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# -----------------------------------------------------------------------
# SECTION 1 – SYNTHETIC DATASET (mirrors real Kaggle dataset structure)
# -----------------------------------------------------------------------
# The real dataset contains district-level dengue case counts per month
# with weather variables (temp, rainfall, humidity).
# We reproduce the same structure synthetically so the code runs
# without a Kaggle API key, but the ANALYSIS LOGIC IS IDENTICAL.

DISTRICTS = [
    "Dhaka", "Chittagong", "Rajshahi", "Sylhet", "Khulna",
    "Barisal", "Rangpur", "Mymensingh", "Comilla", "Narayanganj",
    "Gazipur", "Narsingdi", "Tangail", "Faridpur", "Jessore",
    "Bogra", "Dinajpur", "Pabna", "Sirajganj", "Cox's Bazar"
]

# Simulate monthly dengue case data (2010–2022)
months = pd.date_range("2010-01", "2022-12", freq="MS")
records = []
for district in DISTRICTS:
    base = np.random.randint(20, 500)
    for m in months:
        # Monsoon seasonality – peak June–October
        seasonality = 1 + 2.5 * max(0, np.sin((m.month - 3) * np.pi / 7))
        cases = int(base * seasonality * np.random.lognormal(0, 0.4))
        temp  = np.random.normal(28 + 4 * np.sin((m.month - 1) * np.pi / 6), 1.5)
        rain  = max(0, np.random.normal(120 * seasonality, 40))
        humid = np.random.normal(75, 8)
        records.append({"district": district, "month": m,
                        "cases": cases, "temp_C": round(temp, 1),
                        "rainfall_mm": round(rain, 1),
                        "humidity_pct": round(humid, 1)})

df = pd.DataFrame(records)
print("=" * 60)
print("DENGUE DISEASE SPREAD – SOCIAL NETWORK ANALYSIS")
print("=" * 60)
print(f"\n[DATA] Shape: {df.shape}")
print(df.head())

# -----------------------------------------------------------------------
# SECTION 2 – AGGREGATE & NORMALISE
# -----------------------------------------------------------------------
district_stats = (
    df.groupby("district")
      .agg(total_cases=("cases", "sum"),
           avg_temp=("temp_C", "mean"),
           avg_rain=("rainfall_mm", "mean"),
           avg_humid=("humidity_pct", "mean"))
      .reset_index()
)
district_stats["case_norm"] = (
    district_stats["total_cases"] / district_stats["total_cases"].max()
)
print("\n[STATS] District aggregates (top 5):")
print(district_stats.sort_values("total_cases", ascending=False).head())

# -----------------------------------------------------------------------
# SECTION 3 – BUILD THE TRANSMISSION NETWORK
# -----------------------------------------------------------------------
# Edge weight = composite similarity score (climate + case burden).
# Two districts are linked if their climatic profiles are similar AND
# both have significant case burdens → transmission corridor.

def transmission_weight(r1, r2):
    """
    Compute directed edge weight from district r1 → r2.
    Higher-burden districts 'push' transmission to similar neighbors.
    """
    temp_sim  = 1 / (1 + abs(r1.avg_temp  - r2.avg_temp))
    rain_sim  = 1 / (1 + abs(r1.avg_rain  - r2.avg_rain) / 10)
    humid_sim = 1 / (1 + abs(r1.avg_humid - r2.avg_humid))
    climate_sim = (temp_sim + rain_sim + humid_sim) / 3
    burden_factor = r1.case_norm  # source drives transmission
    weight = climate_sim * burden_factor
    return round(weight, 4)

G = nx.DiGraph()
G.add_nodes_from(DISTRICTS)

for i, r1 in district_stats.iterrows():
    for j, r2 in district_stats.iterrows():
        if i == j:
            continue
        w = transmission_weight(r1, r2)
        if w > 0.25:  # threshold – only meaningful transmission links
            G.add_edge(r1.district, r2.district, weight=w)

print(f"\n[NETWORK] Nodes : {G.number_of_nodes()}")
print(f"[NETWORK] Edges : {G.number_of_edges()}")
print(f"[NETWORK] Density: {nx.density(G):.4f}")

# -----------------------------------------------------------------------
# SECTION 4 – SNA METRICS
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("SNA CENTRALITY METRICS")
print("=" * 60)

# 4.1 Degree Centrality
in_deg  = nx.in_degree_centrality(G)
out_deg = nx.out_degree_centrality(G)

# 4.2 Betweenness Centrality – identifies broker districts
betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

# 4.3 Closeness Centrality – speed of reaching other nodes
closeness = nx.closeness_centrality(G)

# 4.4 PageRank – recursive influence (super-spreader score)
pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

# 4.5 Clustering Coefficient (undirected projection)
G_und = G.to_undirected()
clustering = nx.clustering(G_und, weight="weight")

# Compile into DataFrame
metrics_df = pd.DataFrame({
    "District"       : list(G.nodes()),
    "In_Degree_C"    : [in_deg[n]      for n in G.nodes()],
    "Out_Degree_C"   : [out_deg[n]     for n in G.nodes()],
    "Betweenness"    : [betweenness[n] for n in G.nodes()],
    "Closeness"      : [closeness[n]   for n in G.nodes()],
    "PageRank"       : [pagerank[n]    for n in G.nodes()],
    "Clustering"     : [clustering[n]  for n in G.nodes()],
}).set_index("District").round(4)

metrics_df["TotalCases"] = metrics_df.index.map(
    district_stats.set_index("district")["total_cases"])

print(metrics_df.sort_values("PageRank", ascending=False).to_string())

# 4.6 Top Super-Spreader Districts
top_spreaders = metrics_df["PageRank"].nlargest(5)
print("\n[TOP SUPER-SPREADERS by PageRank]")
for d, v in top_spreaders.items():
    print(f"  {d:20s} PageRank={v:.4f}")

# 4.7 Top Bridge Districts (Betweenness)
top_bridges = metrics_df["Betweenness"].nlargest(5)
print("\n[TOP BRIDGE DISTRICTS by Betweenness Centrality]")
for d, v in top_bridges.items():
    print(f"  {d:20s} Betweenness={v:.4f}")

# 4.8 Global Network Statistics
print("\n[GLOBAL NETWORK STATS]")
print(f"  Number of Strongly Connected Components : {nx.number_strongly_connected_components(G)}")
print(f"  Number of Weakly  Connected Components : {nx.number_weakly_connected_components(G)}")
ug = G.to_undirected()
if nx.is_connected(ug):
    print(f"  Network Diameter       : {nx.diameter(ug)}")
    print(f"  Avg Shortest Path Len  : {nx.average_shortest_path_length(ug):.4f}")
else:
    lcc = max(nx.connected_components(ug), key=len)
    sg  = ug.subgraph(lcc)
    print(f"  LCC Diameter           : {nx.diameter(sg)}")
    print(f"  LCC Avg Path Length    : {nx.average_shortest_path_length(sg):.4f}")

# -----------------------------------------------------------------------
# SECTION 5 – COMMUNITY DETECTION (Greedy Modularity)
# -----------------------------------------------------------------------
communities_gen = nx.community.greedy_modularity_communities(G_und, weight="weight")
communities     = list(communities_gen)
print(f"\n[COMMUNITIES] {len(communities)} epidemic clusters detected")
community_map   = {}
for idx, comm in enumerate(communities):
    print(f"  Cluster {idx+1}: {sorted(comm)}")
    for node in comm:
        community_map[node] = idx

modularity = nx.community.modularity(G_und, communities, weight="weight")
print(f"  Modularity Score: {modularity:.4f}")

# -----------------------------------------------------------------------
# SECTION 6 – VISUALISATIONS (saved as PNG files)
# -----------------------------------------------------------------------
PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
           "#264653", "#8ecae6", "#219ebc"]

def get_node_color(node):
    return PALETTE[community_map.get(node, 0) % len(PALETTE)]

# --- FIG 1: TRANSMISSION NETWORK -----------------------------------------
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

pos   = nx.spring_layout(G, seed=42, k=2.5)
sizes = [5000 * pagerank[n] for n in G.nodes()]
colors= [get_node_color(n) for n in G.nodes()]
edge_w= [G[u][v]["weight"] * 3 for u, v in G.edges()]

nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#ffffff22",
                       width=edge_w, arrows=True,
                       arrowstyle="-|>", arrowsize=15,
                       connectionstyle="arc3,rad=0.1")
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                       node_color=colors, alpha=0.92)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=7,
                        font_color="white", font_weight="bold")

ax.set_title("Dengue Transmission Network – Bangladesh Districts\n"
             "Node size ∝ PageRank | Color = Epidemic Cluster | "
             "Edge weight ∝ Transmission Strength",
             color="white", fontsize=13, pad=15)
ax.axis("off")

patches = [mpatches.Patch(color=PALETTE[i], label=f"Cluster {i+1}")
           for i in range(len(communities))]
ax.legend(handles=patches, loc="lower left", framealpha=0.3,
          labelcolor="white", facecolor="#111")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig1_transmission_network.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\n[SAVED] fig1_transmission_network.png")

# --- FIG 2: CENTRALITY COMPARISON BAR CHARTS ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Centrality Metrics – Dengue Spread Network",
             color="white", fontsize=15, y=1.01)

metrics_plot = metrics_df.sort_values("PageRank", ascending=False)
bar_cfg = [
    ("PageRank",    "#e63946", "PageRank (Super-Spreader Score)"),
    ("Betweenness", "#457b9d", "Betweenness Centrality (Bridge Nodes)"),
    ("Closeness",   "#2a9d8f", "Closeness Centrality (Reach Speed)"),
    ("Clustering",  "#e9c46a", "Clustering Coefficient"),
]
for ax, (col, color, title) in zip(axes.flat, bar_cfg):
    ax.set_facecolor("#161b22")
    sorted_m = metrics_df[col].sort_values(ascending=False)
    bars = ax.barh(sorted_m.index, sorted_m.values, color=color, alpha=0.85)
    ax.set_title(title, color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.set_xlabel("Score", color="#8b949e", fontsize=8)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig2_centrality_metrics.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("[SAVED] fig2_centrality_metrics.png")

# --- FIG 3: SEASONAL TRENDS (Monthly Cases) -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
fig.patch.set_facecolor("#0d1117")

top5 = metrics_df["TotalCases"].nlargest(5).index.tolist()
monthly = df[df["district"].isin(top5)].groupby(["month", "district"])["cases"].sum().reset_index()

ax = axes[0]
ax.set_facecolor("#161b22")
for i, dist in enumerate(top5):
    d = monthly[monthly["district"] == dist]
    ax.plot(d["month"], d["cases"], label=dist,
            color=PALETTE[i], linewidth=1.8, alpha=0.9)
ax.set_title("Monthly Dengue Cases – Top 5 Districts", color="white", fontsize=11)
ax.tick_params(colors="white")
ax.legend(facecolor="#111", labelcolor="white", fontsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

# Monthly heatmap of all districts
ax2 = axes[1]
ax2.set_facecolor("#161b22")
pivot = df.pivot_table(index="district", columns=df["month"].dt.month,
                       values="cases", aggfunc="mean")
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]
im = ax2.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
ax2.set_xticks(range(12))
ax2.set_xticklabels(pivot.columns, color="white", fontsize=8)
ax2.set_yticks(range(len(pivot.index)))
ax2.set_yticklabels(pivot.index, color="white", fontsize=7)
ax2.set_title("Average Cases Heatmap (District × Month)", color="white", fontsize=11)
plt.colorbar(im, ax=ax2, label="Avg Cases")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig3_seasonal_trends.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("[SAVED] fig3_seasonal_trends.png")

# --- FIG 4: DEGREE DISTRIBUTION (Power-law check) -----------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0d1117")

in_degrees  = [d for _, d in G.in_degree()]
out_degrees = [d for _, d in G.out_degree()]

for ax, deg, title, color in zip(
        axes,
        [in_degrees, out_degrees],
        ["In-Degree Distribution", "Out-Degree Distribution"],
        ["#e63946", "#457b9d"]):
    ax.set_facecolor("#161b22")
    ax.hist(deg, bins=10, color=color, edgecolor="#30363d", alpha=0.85)
    ax.set_title(title, color="white", fontsize=11)
    ax.set_xlabel("Degree", color="#8b949e")
    ax.set_ylabel("Frequency", color="#8b949e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

fig.suptitle("Degree Distribution – Dengue Transmission Network",
             color="white", fontsize=13)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fig4_degree_distribution.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("[SAVED] fig4_degree_distribution.png")

# -----------------------------------------------------------------------
# SECTION 7 – EXPORT METRICS TO CSV
# -----------------------------------------------------------------------
metrics_df.to_csv("/mnt/user-data/outputs/sna_metrics.csv")
df.to_csv("/mnt/user-data/outputs/dengue_dataset.csv", index=False)
print("\n[SAVED] sna_metrics.csv")
print("[SAVED] dengue_dataset.csv")
print("\n✅  ALL DONE – check /mnt/user-data/outputs/")
