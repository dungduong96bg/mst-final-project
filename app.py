import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO

st.set_page_config(page_title="MST Clustering App", layout="wide")

st.title("Streamlit — MST clustering interactive app")
st.write("Upload a CSV, choose numeric variables, then build a Minimum Spanning Tree (MST) and cut edges to form clusters.")

# -------------------------
# Upload
# -------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.header("Data preview")
    if st.sidebar.checkbox("Show head", value=True):
        st.dataframe(df.head())

    # choose numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 1:
        st.error("No numeric columns found in the uploaded data. Please upload a dataset with numeric features.")
    else:
        st.sidebar.header("MST options")
        selected = st.sidebar.multiselect("Select variables to use for distance / clustering", numeric_cols, default=numeric_cols[:2])
        standardize = st.sidebar.checkbox("Standardize selected variables (z-score)", value=True)
        metric = st.sidebar.selectbox("Distance metric", ["euclidean", "manhattan", "cosine"], index=0)

        # clustering controls
        cluster_mode = st.sidebar.radio("Cluster by", ("Number of clusters (k)", "Distance threshold"))
        if cluster_mode == "Number of clusters (k)":
            k = st.sidebar.number_input("Number of clusters (k)", min_value=1, max_value=len(df), value=2, step=1)
        else:
            thresh = st.sidebar.number_input("Cut threshold (remove MST edges longer than)", min_value=0.0, value=0.0, step=0.1)

        # additional conditions
        min_obs = st.sidebar.number_input("Minimum observations per cluster", min_value=1, value=1, step=1)

        build = st.sidebar.button("Build MST & cluster")

        if build:
            X = df[selected].copy()
            # drop NA and keep index mapping
            idx_map = X.dropna().index
            X = X.loc[idx_map]

            if standardize:
                Xs = StandardScaler().fit_transform(X.values)
            else:
                Xs = X.values

            # pairwise distances
            D = pairwise_distances(Xs, metric=metric)

            # build full graph
            G_full = nx.Graph()
            n = Xs.shape[0]
            G_full.add_nodes_from(range(n))
            # add weighted edges
            rows, cols = np.triu_indices(n, k=1)
            edges = [(int(i), int(j), float(D[i, j])) for i, j in zip(rows, cols)]
            G_full.add_weighted_edges_from(edges)

            # compute MST
            mst = nx.minimum_spanning_tree(G_full, weight='weight')

            # sort edges by weight
            sorted_edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

            # decide cuts
            if cluster_mode == "Number of clusters (k)":
                if k <= 1:
                    edges_to_remove = []
                else:
                    edges_to_remove = [(u, v) for u, v, d in sorted_edges[:k-1]]
            else:
                edges_to_remove = [(u, v) for u, v, d in mst.edges(data=True) if d['weight'] > thresh]

            mst_cut = mst.copy()
            mst_cut.remove_edges_from(edges_to_remove)

            # clusters
            components = list(nx.connected_components(mst_cut))
            labels = np.empty(n, dtype=int)
            for i, comp in enumerate(components):
                for node in comp:
                    labels[node] = i

            # enforce min_obs filter (clusters smaller than min_obs -> relabel as -1)
            cluster_sizes = pd.Series(labels).value_counts()
            small_clusters = cluster_sizes[cluster_sizes < min_obs].index.tolist()
            for i in range(n):
                if labels[i] in small_clusters:
                    labels[i] = -1

            cluster_series = pd.Series(index=idx_map, data=labels)
            df_result = df.copy()
            df_result = df_result.loc[idx_map]
            df_result['mst_cluster'] = cluster_series.values

            st.success(f"MST built: {n} points, {mst.number_of_edges()} edges. Found {len(set(labels))} clusters (including small/filtered).")

            # Visualization 1: MST graph colored by cluster
            st.subheader("MST graph (layout = spring)")
            fig, ax = plt.subplots(figsize=(8, 6))
            pos = nx.spring_layout(mst, seed=42)

            cmap = plt.get_cmap('tab10')
            unique_labels = np.unique(labels)
            color_map = {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}
            node_colors = [color_map[labels[i]] for i in range(n)]

            nx.draw_networkx_edges(mst, pos, alpha=0.4, ax=ax)
            if edges_to_remove:
                nx.draw_networkx_edges(mst, pos, edgelist=edges_to_remove, style='dashed', edge_color='red', ax=ax)

            nx.draw_networkx_nodes(mst, pos, node_color=node_colors, node_size=60, ax=ax)
            ax.set_axis_off()
            st.pyplot(fig)

            # Visualization 2: scatter or 3D
            st.subheader("Cluster visualization")
            if len(selected) >= 3:
                fig3d = plt.figure(figsize=(8, 6))
                ax3d = fig3d.add_subplot(111, projection='3d')
                for lab in unique_labels:
                    mask = labels == lab
                    ax3d.scatter(Xs[mask, 0], Xs[mask, 1], Xs[mask, 2], label=f'Cluster {lab}')
                ax3d.set_xlabel(selected[0])
                ax3d.set_ylabel(selected[1])
                ax3d.set_zlabel(selected[2])
                ax3d.legend()
                st.pyplot(fig3d)
            elif len(selected) == 2:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                for lab in unique_labels:
                    mask = labels == lab
                    ax2.scatter(Xs[mask, 0], Xs[mask, 1], label=f'Cluster {lab}', s=30)
                ax2.set_xlabel(selected[0])
                ax2.set_ylabel(selected[1])
                ax2.legend()
                st.pyplot(fig2)
            elif len(selected) == 1:
                var = selected[0]
                fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
                for lab in unique_labels:
                    mask = labels == lab
                    ax_hist.hist(X[var].iloc[mask], bins=20, alpha=0.5, label=f'Cluster {lab}')
                ax_hist.set_xlabel(var)
                ax_hist.set_ylabel("Frequency")
                ax_hist.legend()
                st.pyplot(fig_hist)
            else:
                pca = PCA(n_components=3)
                proj = pca.fit_transform(Xs)
                fig3d = plt.figure(figsize=(8, 6))
                ax3d = fig3d.add_subplot(111, projection='3d')
                for lab in unique_labels:
                    mask = labels == lab
                    ax3d.scatter(proj[mask, 0], proj[mask, 1], proj[mask, 2], label=f'Cluster {lab}')
                ax3d.set_xlabel('PCA1')
                ax3d.set_ylabel('PCA2')
                ax3d.set_zlabel('PCA3')
                ax3d.legend()
                st.pyplot(fig3d)

            # show cluster sizes
            st.subheader("Cluster sizes")
            sizes = pd.Series(labels).value_counts().sort_index()
            st.table(pd.DataFrame({'cluster': sizes.index, 'size': sizes.values}))

            # allow download of labeled csv
            buffer = BytesIO()
            df_result.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button("Download labeled CSV", data=buffer, file_name="mst_labeled.csv", mime='text/csv')

            # show sample of labeled data
            st.subheader("Labeled data sample")
            st.dataframe(df_result.head(200))

        else:
            st.info("Select variables and press 'Build MST & cluster' in the sidebar.")

else:
    st.info("Upload a CSV to begin. Example: dataset with numeric features (x,y,...) or any tabular data.")

# -------------------------
# Footer / tips
# -------------------------
st.markdown("---")
st.markdown("**Tips:**\n- MST clustering: to get k clusters, remove the k-1 largest edges in the MST.\n- If your data has many points (>2000), building the full dense graph may be slow or memory heavy. Consider sampling or using approximate neighbor graphs.\n- 3D plots appear if you select 3 or more variables (or PCA fallback). Use mouse to rotate the 3D plot.")
