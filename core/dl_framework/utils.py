from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

    fig = make_subplots(rows=1, cols=len(plot_dict), subplot_titles=list(plot_dict.keys()))

    for idx, (met, mets) in enumerate(plot_dict.items()):
        for key in mets:
            fig.add_trace(go.Scatter(x=history.index, y=history[key].values, name=key), row=1, col=idx+1 )

    fig.show()
