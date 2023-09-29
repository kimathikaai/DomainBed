import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.cm as cm
import os
import numpy as np
from glob import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
plt.rcParams["font.family"] = "Times New Roman"

def get_pickle_files(folders, base_path):
    if not isinstance(folders, list):
        folders = [folders]
    files = []
    for folder in tqdm(folders):
        path = os.path.join(base_path, folder, "*.pickle")
        # print(path)
        # path = f'/Users/noname/scratch/saved/domainbed_results/tsne/{folder}/*.pickle'
        path = sorted(glob(path, recursive=True))[0]
        assert os.path.exists(path), path
        files.append(path)
    return files

def get_tsne_df_list(files):
    if not isinstance(files, list):
        files = [files]
    df_list = []
    for file in tqdm(files):
        # print('[processing]: ', file)
        df_list.append((get_tsne_df(file), file))
    return df_list

def get_tsne_df(path, pca_components=48):

    df = pd.read_pickle(path)

    pca = PCA(n_components=pca_components)
    zs = np.array(list(df['latent_vector']))
    pca.fit(zs)
    # print('Cumulative explained variation for {} principal components: {}'.format(
    #     pca_components, np.sum(pca.explained_variance_ratio_)))

    # use tsne
    zs = pca.transform(zs)
    # tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    tsne = TSNE(n_components=2, perplexity=35)
    df['tsne_embeddings'] = list(tsne.fit_transform(zs))

    return df

def camera_plot_embeddings(data, ax, point_type, colors, alpha=1, s=1.5, edgecolor=None):
    #markers = ['o', 'v', 'p', '*', 'D', '>', 'P']
    labels = sorted(data[point_type].unique())
    for i, label in enumerate(labels):
        ax.scatter(
            x=np.array(list(data[data[point_type]==label]['tsne_embeddings']))[:, 0].tolist(), 
            y=np.array(list(data[data[point_type]==label]['tsne_embeddings']))[:, 1].tolist(), 
            s=mpl.rcParams['lines.markersize'] ** s,
            color = colors[i], 
            label=label,
            marker='.',
            alpha=alpha,
            edgecolor=edgecolor
            )
    #ax.axis('off')
    # ax.axes.get_xaxis().set_ticklabels([])
    # ax.axes.get_yaxis().set_ticklabels([])
    
    return ax

def plot_embeddings(df_list):
    """Plot source class, source domains, target class"""
    if not isinstance(df_list, list):
        df_list = [df_list]

    print("Number of plots: ", len(df_list))
        
    
    domain_colors = cm.Blues(np.linspace(0,1, len(df_list[0][0]["domain"].unique())+1)[1:])
    class_colors = list(mcolors.TABLEAU_COLORS.keys())

    class_legend = None
    domain_legend = None
    fig = None
    
    for i, (df, file) in enumerate(df_list):
        # GET TEST DOMAIN
        test_env = df[df['is_test'] == 1]['domain'].unique()
        assert len(test_env) == 1
        test_env = test_env[0]

        # GET FIGURE AND AXES
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(14,3), sharey=True)
        fig.suptitle(f"Test Domain: {test_env}\nPath: {file}", fontsize=15)
        fig.tight_layout()
        
        # SOURCE DOMAIN CLASSES
        point_type = 'class'
        data = df.loc[(df['is_test']==0)].sort_values(by=point_type)
        camera_plot_embeddings(data=data, ax=ax[0], point_type=point_type, colors=class_colors)
        # ax[0].set_title("Source: Class Embeddings", fontsize=20)

        # SOURCE DOMAINS
        point_type = 'domain'
        data = df.loc[(df['is_test']==0)].sort_values(by=point_type)
        camera_plot_embeddings(data=data, ax=ax[1], point_type=point_type, colors=domain_colors)
        # ax[1].set_title("Source: Domain Embeddings", fontsize=20)
        # ax_legend = ax[1].get_legend_handles_labels()

        # TARGET DOMAIN CLASSES
        point_type = 'class'
        data = df.loc[(df['is_test']==1)].sort_values(by=point_type)
        camera_plot_embeddings(data=data, ax=ax[2], point_type=point_type, colors=class_colors)    
        # ax[2].set_title("Target: Class Embeddings", fontsize=20)

        # return fig, ax
        # ax[2].legend(
        #     title="Classes", 
        #     markerscale=2, 
        #     loc='center right', 
        #     # bbox_to_anchor=(1.15,0.5),
        #     # frameon=False
        # )

        # # LEGEND
        # domain_legend = ax[1].get_legend_handles_labels()
        # class_legend = ax[0].get_legend_handles_labels()
        #
        # # get non-overlapping class legends
        # noc_legend = ([],[])
        # oc_legend = ([],[])
        # for handle, label in zip(class_legend[0], class_legend[1]):
        #     if label in ['1', '6']:
        #         noc_legend[0].append(handle)
        #         noc_legend[1].append(label)
        #     else:
        #         oc_legend[0].append(handle)
        #         oc_legend[1].append(label)
        #
        # # Create a Non-overlapping class legend
        # noc_legend = fig.legend(
        #     title="$C_N$",
        #     handles=noc_legend[0], 
        #     #labels=noc_legend[1], 
        #     labels=[None]*len(noc_legend[1]),
        #     markerscale=5,
        #     loc='upper left', 
        #     ncols=len(noc_legend[0]),
        #     bbox_to_anchor=(0.55,0),
        #     columnspacing=0.2,
        #     title_fontsize=20,
        #     #bbox_to_anchor=(1,0.65),
        #     )
        # # Add the legend manually to the current Axes.
        # ax = plt.gca().add_artist(noc_legend)
        # # Create a Overlapping class legend
        # oc_legend = fig.legend(
        #     title="$C_O$",
        #     handles=oc_legend[0], 
        #     #labels=oc_legend[1],
        #     labels=[None]*len(oc_legend[1]),
        #     markerscale=5,
        #     loc='upper left', 
        #     ncols=len(oc_legend[0]),
        #     bbox_to_anchor=(0.2,0),
        #     columnspacing=0.2,
        #     title_fontsize=20,
        #     #bbox_to_anchor=(1,0.5),
        #     )
        # # Add the legend manually to the current Axes.
        # ax = plt.gca().add_artist(oc_legend)
        #
        # # Create a domain legend
        # fig.legend(
        #     title="Source Domains",
        #     handles=domain_legend[0], 
        #     #labels=ax_legend[1][-3:], 
        #     labels=['','',''],
        #     loc='upper left', 
        #     ncols=len(domain_legend[0]), 
        #     markerscale=5,
        #     bbox_to_anchor=(0,0),
        #     columnspacing=0.2,
        #     title_fontsize=20,
        #     #bbox_to_anchor=(1,0.35),
        #     )











