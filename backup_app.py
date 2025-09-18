import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import load_iris

st.set_page_config(page_title="MST Interactive Clustering", layout="wide")

# ==== Load Iris ====
iris = load_iris()
X_full = iris.data
feature_names = iris.feature_names
n_points = X_full.shape[0]

# ==== Sidebar: chọn feature ====
st.sidebar.header("Chọn chiều dữ liệu")
f1 = st.sidebar.selectbox("Chiều X", range(X_full.shape[1]), format_func=lambda i: feature_names[i])
f2 = st.sidebar.selectbox("Chiều Y", range(X_full.shape[1]), index=1, format_func=lambda i: feature_names[i])
X = X_full[:, [f1, f2]]

# ==== Sidebar: chọn số cụm k ====
st.sidebar.header("Cấu hình MST & clustering")
k_clusters = st.sidebar.selectbox("Chọn số cụm k", [0,1,2,3,4,5,6,7,8,9,10])

# ==== Build full MST ====
G = nx.Graph()
for i in range(n_points):
    for j in range(i+1, n_points):
        dist = np.linalg.norm(X[i]-X[j])
        G.add_edge(i, j, weight=dist)

MST = nx.minimum_spanning_tree(G, algorithm='kruskal')
edges_sorted = sorted(MST.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

# ==== Cut MST to form clusters ====
edges_to_remove = edges_sorted[:k_clusters-1]
G_cut = MST.copy()
G_cut.remove_edges_from([(u,v) for u,v,_ in edges_to_remove])

# ==== Find connected components ====
clusters = list(nx.connected_components(G_cut))
cluster_colors = np.zeros(n_points)
for idx, comp in enumerate(clusters):
    for node in comp:
        cluster_colors[node] = idx

# ==== Plot MST + clusters ====
fig, ax = plt.subplots(figsize=(8,6))
pos = {i: X[i] for i in range(n_points)}

# Vẽ toàn bộ MST (màu nhạt)
nx.draw_networkx_edges(MST, pos, ax=ax, edge_color="yellow", style="dashed")

# Vẽ các cạnh MST sau khi cắt
nx.draw_networkx_edges(G_cut, pos, ax=ax, edge_color="gray")

# Vẽ các node theo cụm
ax.scatter(X[:,0], X[:,1], c=cluster_colors, cmap=plt.cm.Set1, s=50, edgecolors='k')

ax.set_xlabel(feature_names[f1])
ax.set_ylabel(feature_names[f2])
ax.set_title(f"MST cắt {k_clusters-1} cạnh dài nhất → {k_clusters} cụm")

ax.axis('equal')
plt.tight_layout()

st.pyplot(fig)

# ==== Show clusters info ====
st.markdown("### Các cụm sau khi cắt MST")
for idx, comp in enumerate(clusters):
    st.write(f"Cụm {idx+1}: {sorted(list(comp))}")



