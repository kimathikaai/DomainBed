import re
import pathlib
import pandas as pd
from typing import List
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import itertools

METHOD_NAME = "FOND"

VIABLE_SELECTION_METRICS = ["acc", "f1", "oacc", "nacc", "macc", "vacc"] + [
    "accC" + str(i) for i in range(65)
]
VIABLE_EVALUATION_METRICS = VIABLE_SELECTION_METRICS

BASELINES = [
    "SelfReg",
    "MLDG",
    "Transfer",
    "ERM",
    "CAD",
    "ARM",
    "CORAL",
    # "CausIRL_MMD",
    # "CausIRL_CORAL",
    "CausIRL",
    "PGrad",
    "EQRM",
]
METHODS = [METHOD_NAME]

RENAMES = {
    # "XDomError": METHOD_NAME + r"\B",
    "XDom": METHOD_NAME + r"\FB",
    "XDomBeta": METHOD_NAME + r"\F",
    "XDomBetaError": METHOD_NAME,
    "SupCon": METHOD_NAME + r"\FBA",
    "CausIRL_CORAL": "CausIRL"
}

ABLATIONS = [
    METHOD_NAME + r"\FB",
    METHOD_NAME + r"\F",
    METHOD_NAME,
    METHOD_NAME + r"\FBA",
    # "NOC"
]

# ABLATIONS = list(RENAMES.values()).remove('CausIRL')

AXIS_LABELS = {
    "nacc": r"Domain-Linked ($\mathcal{Y}_{L}$) Accuracy",
    "oacc": r"Domain-Shared ($\mathcal{Y}_{S}$) Accuracy",
    "av_acc": r"Average ($C_{O},C_{N}$) Accuracy",
    "macc": "Macro-Class Accuracy",
    "vacc": "rAverage ($C_{O},C_{N}$) Accuracy",
    "f1": "F1-Score",
    "33": "low",
    "66": "high",
}

# BASELINE_MARKERS = ["o"]
ABLATION_MARKERS = ["D", "p", "v", "X", "o"]
BASELINE_MARKERS = [
    ">",
    "o",
    "s",
    "P",
    "*",
    "H",
    "d",
    "<",
    "h",
    "8",
    "^",
    # "4",
    "$\\clubsuit$",
]

# base_colors = cm.Set3(np.linspace(0, 1, len(BASELINES)))
# base_colors = cm.YlOrRd(np.linspace(0, 1, len(BASELINES)))
# base_colors = list(mcolors.TABLEAU_COLORS.keys())
BASE_COLORS = [
    "red",
    "darkviolet",
    "orange",
    "cornflowerblue",
    "limegreen",
    "yellow",
    "darkgrey",
    "pink",
    'blue',
    'limegreen',
    'magenta'
]


MARKERS = BASELINE_MARKERS + ABLATION_MARKERS


# file scraping function that returns a row
# Inputs: file_path
# Returns: row as defined by ROW_TEMPLATE (perform check in function)
def scrape_latex(latex_file_path) -> List[dict]:
    rows = []

    def get_header(line):
        header = re.split(
            "\s+",
            re.sub(
                r"\\textbf{(\w+)}", r"\1", line.replace("&", "").replace(r"\\", "")
            ).strip(),
        )
        dataset_list = header[1:-1]
        return header, dataset_list

    # get selection and evaluation metric
    file_name = pathlib.Path(latex_file_path).stem
    selection_metric = file_name.split("_")[-2]
    evaluation_metric = file_name.split("_")[-1]
    overlap = file_name.split("_")[-3]

    # print(file_name, selection_metric, evaluation_metric, overlap)
    assert selection_metric in VIABLE_SELECTION_METRICS
    assert evaluation_metric in VIABLE_EVALUATION_METRICS

    # read file contents
    with open(latex_file_path, "r") as f:
        found_section = False
        found_table = False
        found_header = False
        found_table_start = False

        lines = f.readlines()
        for row_n, line in enumerate(lines):
            # print(row_n,line)
            # get to training-domain model selection section
            if re.search("subsection{Model.*training-domain", line) or found_section:
                # if not found_section: print("*"*10, line)
                found_section = True
                if re.search("subsubsection{Averages}", line) or found_table:
                    # if not found_table: print("*"*10, line)
                    found_table = True
                    if re.search("textbf{Algorithm}", line) or found_header:
                        if not found_header:
                            # therefore first time to find header
                            # get header items
                            header, dataset_list = get_header(line)
                            # print(header)
                            # print(dataset_list)

                            found_header = True

                        # look for table beginning
                        if "midrule" in line or found_table_start:
                            if not found_table_start:
                                found_table_start = True
                                continue

                            # don't process after /bottom/rule
                            if "bottomrule" in line:
                                # finished reading table
                                break

                            # strip the row for each algorithm for
                            # value and std per dataset
                            algo_row = re.split(
                                "\s+",
                                line.replace("$\\pm$ ", "")
                                .replace("&", "")
                                .replace(r"\\", "")
                                .strip(),
                            )
                            algorithm = algo_row.pop(0)
                            average = algo_row.pop()
                            values = algo_row
                            # print("*"*10, algo_row)

                            # Support 'X'
                            org_values = values[:]
                            values = []
                            for item in org_values:
                                if item == "X":
                                    values.extend(["X", "X"])
                                else:
                                    values.append(item)

                            for idx, dataset in enumerate(dataset_list):
                                if values[idx * 2] == "X":
                                    continue
                                if float(values[idx * 2]) < 0:
                                    continue
                                if algorithm in RENAMES:
                                    algorithm = RENAMES[algorithm]

                                row = {
                                    "dataset": dataset,
                                    "overlap": overlap,
                                    "algorithm": algorithm,
                                    "selection_metric": selection_metric,
                                    "evaluation_metric": evaluation_metric,
                                    "selection_value": None,
                                    "evaluation_value": float(values[idx * 2]),
                                    "selection_std": None,
                                    "evaluation_std": float(values[idx * 2 + 1]),
                                    "baseline": 1 if algorithm in BASELINES else 0,
                                }
                                rows.append(row)
    return rows


import math
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator


def get_metrics(
    df, eval_metric, selec_metric, dataset=None, overlap=None, algorithm=None
):
    # print(f"({selec_metric}, {eval_metric}, {dataset})")
    OVERLAPS = ["33", "66"]
    data = df.loc[
        (df["selection_metric"] == selec_metric)
        & (df["evaluation_metric"] == eval_metric)
    ]
    if algorithm is not None:
        if not isinstance(algorithm, list):
            algorithm = [algorithm]
        data = data.loc[data["algorithm"].isin(algorithm)]
    else:
        data = data.loc[data["algorithm"].isin(BASELINES + ABLATIONS)]

    if dataset is not None:
        if not isinstance(dataset, list):
            dataset = [dataset]
        data = data.loc[data["dataset"].isin(dataset)]

    if overlap is not None:
        if not isinstance(overlap, list):
            overlap = [overlap]
        data = data.loc[data["overlap"].isin(overlap)]
    else:
        data = data.loc[data["overlap"].isin(OVERLAPS)]

    data = data.sort_values(by=["baseline", "algorithm"], ascending=False).reset_index(
        drop=True
    )
    # cols = ["algorithm", "overlap", "dataset", "evaluation_value"]
    metric_data = (
        data.groupby("algorithm").mean(numeric_only=True).reset_index(names="algorithm")
    )
    return metric_data


def plot_overlap_dataset(
    df,
    ax,
    selec_metric,
    x,
    y,
    color_metric=None,
    dataset=None,
    overlap=None,
    markersize=2.75,
    errorbar=True,
    plot_dots=True,
):
    """Plot average across overlaps for each dataset"""
    baseline = get_metrics(df, "oacc", selec_metric, dataset=dataset, overlap=overlap)[
        "baseline"
    ]
    algorithm = get_metrics(df, "oacc", selec_metric, dataset=dataset, overlap=overlap)[
        "algorithm"
    ]
    df_oacc = get_metrics(df, "oacc", selec_metric, dataset=dataset, overlap=overlap)
    df_nacc = get_metrics(df, "nacc", selec_metric, dataset=dataset, overlap=overlap)
    df_acc = get_metrics(df, "acc", selec_metric, dataset=dataset, overlap=overlap)[
        "evaluation_value"
    ]
    df_macc = get_metrics(df, "macc", selec_metric, dataset=dataset, overlap=overlap)
    df_f1 = get_metrics(df, "f1", selec_metric, dataset=dataset, overlap=overlap)[
        "evaluation_value"
    ]
    # df_diff = df_oacc - df_nacc
    # df_ave_acc = (df_oacc + df_nacc) / 2
    assert len(df_oacc) == len(df_nacc) == len(df_acc) == len(algorithm)

    assert len(MARKERS) >= len(algorithm)

    # dataframe
    data = (
        pd.DataFrame(
            data={
                "marker": MARKERS[: len(algorithm)],
                "algorithm": algorithm,
                "nacc": df_nacc["evaluation_value"],
                "nacc_err": df_nacc["evaluation_std"],
                "oacc": df_oacc["evaluation_value"],
                "oacc_err": df_oacc["evaluation_std"],
                "acc": df_acc,
                # "av_acc": df_ave_acc,
                "f1": df_f1,
                "macc": df_macc["evaluation_value"],
                "macc_err": df_macc["evaluation_std"],
                # "diff": df_diff,
                "baseline": baseline,
            }
        )
        .sort_values(by=["baseline", "algorithm"], ascending=[False, True])
        .reset_index(drop=True)
    )
    # print(data)

    # colour mapping
    my_cmap = plt.get_cmap("viridis")

    def rescale(row, rows):
        return (row - np.min(rows)) / (np.max(rows) - np.min(rows))

    i_p = 0
    i_b = 0
    # populate scatter plot
    for index, row in data.iterrows():
        if row["algorithm"] in ABLATIONS:
            color = "cyan"
            marker = ABLATION_MARKERS[i_p]
            i_p += 1
        else:
            color = BASE_COLORS[i_b]
            marker = BASELINE_MARKERS[i_b]
            i_b += 1

        if errorbar:
            ax.errorbar(
                x=row[x],
                y=row[y],
                yerr=row[y + "_err"],
                xerr=row[x + "_err"],
                color="grey",
                zorder=5,
                elinewidth=0.5,
                capsize=2,
            )
        ax.scatter(
            x=row[x],
            y=row[y],
            # color=my_cmap(rescale(row[color_metric], data[color_metric])),
            color=color,
            edgecolor="black",
            label=row["algorithm"],
            marker=marker,
            s=mpl.rcParams["lines.markersize"] ** markersize,
            zorder=10,
        )
        # black dot for visualization
        if plot_dots:
            ax.scatter(
                x=row[x],
                y=row[y],
                color="black",
                label="_nolegend_",
                marker=".",
                s=mpl.rcParams["lines.markersize"] ** 1,
                zorder=15,
            )

    return data


def stack_plot_results(df, selec_metric, eval_metric):
    dataset_list = ["PACS", "VLCS", "OfficeHome"]
    overlap_list = ["33", "66"]
    color = ["#024b7a", "#44a5c2"]

    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(dataset_list),
        # figsize=(10,10),
        figsize=(10, 4),
        sharey=False,
    )

    for i, dataset in enumerate(dataset_list):
        data = df.loc[
            (df["dataset"] == dataset)
            & (df["selection_metric"] == selec_metric)
            & (df["evaluation_metric"] == eval_metric)
            & (df["algorithm"] != "SupCon")
            &
            # (df['algorithm'] != 'Intra') &
            # (df['algorithm'] != 'Intra_XDom') &
            (df["algorithm"] != "XDomBatch")
        ].sort_values(by=["baseline", "algorithm"], ascending=[False, True])

        values = []
        for overlap in overlap_list:
            values.append(
                list(data.loc[(df["overlap"] == overlap)]["evaluation_value"])
            )
        values = np.array(values)

        # set_trace()
        # stack bar charts
        for j in range(values.shape[0]):
            ax[i].bar(
                x=list(data[data["overlap"] == overlap_list[j]]["algorithm"].unique()),
                height=values[j],
                bottom=np.sum(values[:j], axis=0),
                color=color[j],
                label=overlap_list[j],
            )
        ax[i].set_title(dataset)
        ax[i].set_ylabel = selec_metric
        ax[i].tick_params(axis="x", labelrotation=90)
        ax[i].grid(axis="y", which="both")
        ax[i].yaxis.set_minor_locator(AutoMinorLocator(2))

        if i == 0:
            handles, labels = ax[i].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                title="Overlap Cases",
                ncols=len(overlap_list),
                bbox_to_anchor=(0.5, 0.93),
            )

    fig.suptitle(f"(s,e) = ({selec_metric},{eval_metric})")
    # fig.legend(loc="upper center", title="Overlap Case", ncols=2)
    fig.tight_layout(pad=3.0, w_pad=1.0)
    return fig


# sl = list(df["selection_metric"].unique())
# el = list(df["evaluation_metric"].unique())
# sl_el = itertools.product(sl, el)
# for s, e in sl_el:
#     #if e != 'nacc' or s != 'nacc': continue
#     #if s != 'nacc': continue
#     if s != e: continue
#     #stack_plot_results(df=df, selec_metric=s, eval_metric=e).show()
def plot_results(df, selec_metric, eval_metric, overlap_list, dataset_list):
    my_cmap = plt.get_cmap("viridis")

    # rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    # rescale = lambda y: max((y - np.mean(y)) / (np.max(y) - np.mean(y)), 0)
    def rescale(y):
        # zero = np.mean(y)
        zero = sorted(y, reverse=True)[len(y) // 3]
        # everything above 2 quadrant is colored
        y = np.array((y - zero) / (np.max(y) - zero))
        _y = y <= 0
        _y = np.where(_y)[0]
        y[_y] = 0

        return y

    fig, ax = plt.subplots(
        nrows=len(dataset_list),
        ncols=len(overlap_list),
        figsize=(8, 10),
        # figsize=(8,5),
        sharey=False,
    )

    if len(dataset_list) == 1:
        ax = [ax]

    for i, dataset in enumerate(dataset_list):
        for j, overlap in enumerate(overlap_list):
            data = df.loc[
                (df["dataset"] == dataset)
                & (df["selection_metric"] == selec_metric)
                & (df["evaluation_metric"] == eval_metric)
                & (df["overlap"] == overlap)
            ].sort_values(by=["baseline", "algorithm"], ascending=[False, True])

            ax[i][j].bar(
                data.algorithm,
                data.evaluation_value,
                color=my_cmap(rescale(data.evaluation_value)),
                width=0.6,
            )

            # plot_data = df_group.mean()['evaluation_value']
            # max_algo = plot_data.idxmax()
            # print(plot_data)

            # plot_data.plot(ax=ax[i][j], kind="bar")
            ax[i][j].set_title(f"{dataset}-{overlap}")
            # ax[i][j].set_yticks(np.arange(20,60,5))
            ax[i][j].yaxis.set_minor_locator(AutoMinorLocator(2))
            ax[i][j].get_yaxis().set_major_locator(MaxNLocator(integer=True))
            ax[i][j].set_ylim(
                min(data.evaluation_value) - 3, max(data.evaluation_value) + 3
            )
            # ax[i][j].set_ylim(40, 60)
            ax[i][j].grid(axis="y", which="both")
            ax[i][j].tick_params(axis="x", labelrotation=90)
            # ax[i][j].legend()

    fig.suptitle(f"(s,e) = ({selec_metric},{eval_metric})")
    fig.tight_layout(pad=1.0, h_pad=2.0)
    return fig
