from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
import pytz
from pathlib import Path
import os
import pandas as pd


def read_config(config_file):
    setup_config = {}
    # abr_dict = {}
    for key, value in config_file.items():
        if key == "title":
            continue
        else:
            for subkey, subval in value.items():
                if type(subval) != dict:
                    key_name = f"{key[0]}_{subkey}"
                    setup_config[key_name] = subval
                else:
                    for subsubkey, subsubval in subval.items():
                        key_name = f"{key[0]}_{subkey[0]}_{subsubkey}"
                        setup_config[key_name] = subsubval
    return setup_config


def plot_history(history):
    implemented_metric = ["acc", "loss"]
    plot_dict = {}
    for met in implemented_metric:
        plot_dict[met] = [s for s in history if met in s]

    fig = make_subplots(rows=1, cols=len(plot_dict),
                        subplot_titles=list(plot_dict.keys()))

    for idx, (met, mets) in enumerate(plot_dict.items()):
        for key in mets:
            fig.add_trace(go.Scatter(x=history.index,
                          y=history[key].values, name=key), row=1, col=idx+1)

    fig.show()


def get_history(ckp_path, monitor, fileformat=["csv"]):
    project_name = Path(os.getcwd()).name
    for i, path in enumerate(ckp_path.parts):
        length = len(ckp_path.parts)
        if (path == project_name) and (i+1 != length):
            project_path = ckp_path.parents[length-i-2]

    files = []
    if type(fileformat) != list:
        fileformat = [fileformat]
    for fmt in fileformat:
        for rel_path in ckp_path.rglob("*."+fmt):
            # files.append(rel_path.relative_to(project_path))
            files.append(rel_path)
        metric_files = []
        for metric in files:
            if monitor in metric.name:
                metric_files.append(metric)

    return metric_files


def get_specific_history(ckp_path, monitor, fileformat=["csv"], specific="best"):
    files = get_history(ckp_path, monitor, fileformat)
    if specific == "best":
        best_vals = []
        for idx, hist in enumerate(files):
            get_func = getattr(pd, "read_" + hist.suffix[1:])
            tmp_df = get_func(hist)
            if "acc" in monitor:
                best_vals.append(tmp_df[monitor].max())
                best_idx = best_vals.index(max(best_vals))
            if "loss" in monitor:
                best_vals.append(tmp_df[monitor].min())
                best_idx = best_vals.index(min(best_vals))
        return getattr(pd, "read_" + files[best_idx].suffix[1:])(files[best_idx]), files[best_idx]

    if specific == "all":
        df_list = []
        for idx, hist in enumerate(files):
            get_func = getattr(pd, "read_" + hist.suffix[1:])
            tmp_df = get_func(hist)
            df_list.append(tmp_df)
        # best_idx = best_vals.index(max(best_vals))
        # best_hist =
        return df_list
