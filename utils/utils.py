import base64
import os

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from math import ceil
from plotly.subplots import make_subplots
from collections import deque
from scipy.stats import skew, kurtosis

import h5py
import importlib
import utils.SK_evaluation as evlSK
importlib.reload(evlSK)
matplotlib.use('Agg')

def create_graph_config():
    return {
        'displayModeBar': True,
        'scrollZoom': False,
        'editable': True,
        'edits': {
        'shapePosition': True,  # Enable shape dragging
        },
    }

def create_graph_layout():
    return {
        'dragmode': 'draggable',
        'hovermode': 'closest'
    }

def create_graph_config_general(num_group_subplots, file_names):

    # Create figures
    fig_area_all = make_subplots(
        rows=num_group_subplots, cols=1, 
        subplot_titles=tuple(file_names),
        vertical_spacing=0.045,
    )
    fig_area_all.update_layout(
        title_text="<b>Area under the radius curve for all runs in the group<b>",
        title_x=0.5,
        dragmode=False,
        height=300*num_group_subplots,
        showlegend=True,
        legend_tracegroupgap=2400./num_group_subplots - 180.,
    )
    fig_mov_avg_all = make_subplots(
        rows=num_group_subplots, cols=1, 
        subplot_titles=tuple(file_names),
        vertical_spacing=0.045,
    )
    fig_mov_avg_all.update_layout(
        title_text="<b>Moving average for all runs in the group<b>",
        title_x=0.5,
        dragmode=False,
        height=300*num_group_subplots,
        showlegend=True,
        legend_tracegroupgap=2400./num_group_subplots - 139.,
    )
    return fig_area_all, fig_mov_avg_all

def create_graph_config_features(num_group_subplots, file_names, num_graphs):
    # Create figures
    fig_features_all = {}
    for i in range(1, num_graphs + 1):
        fig = make_subplots(
            rows=ceil(num_group_subplots / 3), cols=3, 
            subplot_titles=tuple(file_names),
            vertical_spacing=0.13,
        )
        fig.update_layout(
            title_x=0.5,
            dragmode=False,
            height=100*num_group_subplots,
            showlegend=False,
            # legend_tracegroupgap=2400./num_group_subplots - 180.,
        )
        fig_features_all[f"{i}"] = fig
    return fig_features_all

def save_file(name, content, upload_directory):
    """
    Encode data to base64 and save it to the specified upload directory
    """
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(upload_directory, name), "wb") as fp:
        # fp.write(base64.decodebytes(data))
        fp.write(base64.b64decode(data))

def get_uploaded_files(upload_directory):
    """
    Get uploaded files in the specified upload directory
    """
    files = []
    for filename in os.listdir(upload_directory):
        path = os.path.join(upload_directory, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

def load_dataframe_hdf5(filepath):
    """
    Read hdf5 file
    """
    data_arrays = {}
    with h5py.File(filepath, "r") as hdf:
        test_drives_group = hdf["TestDrives"]
        for drive_name in test_drives_group.keys():
            drive_group = test_drives_group[drive_name]  # Access the drive group
            data_dict = {}
            for column_name in drive_group.keys():
                column_data = drive_group[column_name][:]
                if column_data.dtype.names: 
                    column_data = column_data[column_data.dtype.names[0]]
                data_dict[column_name] = column_data
            data_arrays[test_drives_group[drive_name].attrs["file_name"]] = pd.DataFrame(data_dict)
    return data_arrays

def load_metadata(filepath):
    """
    Read metadata from hdf5 file
    """
    try:
        with h5py.File(filepath, "r") as hdf:
            dataset = hdf["Configuration"]["Metadata"]
            headers = dataset.dtype.names
            data = list(dataset[()].tolist())
            headers = [header.decode('utf-8') if isinstance(header, bytes) else header for header in headers]
            return pd.DataFrame([data], columns=headers)
    except KeyError:
        raise ValueError(f"The dataset does not exist in the file '{filepath}'.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the HDF5 file: {e}")


def load_triggers(rawDataDir):
    with h5py.File(rawDataDir, "r") as hdf:
        triggers_folder = hdf["Configuration"]["Maneuver Specification"]["Triggers"]
        start = np.array(triggers_folder["Start"])
        stop = np.array(triggers_folder["Stop"])
    start = 100
    stop = 9000
    df = pd.DataFrame(data=np.array([[start, stop]]), columns=['start', 'stop'])
    return df

def delete_file(filename, upload_directory):
    """
    Delete file
    """
    if os.path.isfile(os.path.join(upload_directory, filename)):
        os.remove(os.path.join(upload_directory, filename))

def transform_data(data):
    """
    Transform data using PCA
    """
    pca = PCA(n_compon9ents=len(data), whiten=False)
    
    data = np.asarray(data).T
    pca.fit(data)
    transform_data = pca.transform(data)

    return transform_data

def extract_data(df_list, triggers, limits):
    """
    Extract dataframe and store them in a dictionary
    """
    all_data = {}
    for tab in df_list:
        datatable = pd.DataFrame(tab['props']['children'][0]['props']['data'])
        file_name = tab['props']['value']
        all_data[file_name] = SK_radius_eval(datatable, triggers, limits)
    return all_data

def add_trigger_lines(fig, row_num, triggers, r):
    """Helper function to add trigger lines consistently"""
    shapes = []

    # Add vertical lines with drag capability
    v_lines = [
        (triggers['start'], 'Start trigger', 'green', True),
        (triggers['stop'], 'Stop trigger', 'red', True)
    ]
    for x_val, name, color, show_legend in v_lines:
        # Add a trace for the legend
        fig.add_trace(go.Scatter(
            x=[x_val],
            y=[40],  # Middle of plot
            mode='lines',
            name=name,
            line=dict(color=color, width=3, dash='dashdot'),
            showlegend=show_legend,
            legendgroup=str(row_num),
        ), row=row_num, col=1)

        # Add the vertical line as a shape
        shapes.append({
            "type": "line",
            "x0": x_val, 
            "x1": x_val, 
            "y0": 0, 
            "y1": 80,
            "xref": f"x{row_num}", 
            "yref": f"y{row_num}",
            "line": {"color": color, "width": 3, "dash": "dashdot"},
            "editable": True,  # Make line draggable
        })
    return shapes

def add_limit_functions(fig, row_num, signal_len, subplot_limits):
    """Helper to change function limits consistently"""
    shapes = []

    h_lines = [
        (subplot_limits['alarm_low'], 'Alarm low', 'red', True),
        (subplot_limits['alarm_high'], 'Alarm high', 'red', False),
        (subplot_limits['warning_low'], 'Warning low', 'green', True),
        (subplot_limits['target'], 'Target', 'black', True),
        (subplot_limits['warning_high'], 'Warning high', 'green', False)
    ]
    for y_val, name, color, show_legend in h_lines:
        # Add a trace for the legend
        fig.add_trace(go.Scatter(
            x=[signal_len], #[-5000, signal_len + 5000],
            y=[y_val], #* signal_len,
            mode='lines',
            name=name,
            line=dict(color=color, width=3, dash='dash'),
            showlegend=show_legend,
            legendgroup=str(row_num),
        ), row=row_num, col=1)

        # Add the vertical line as a shape
        shapes.append({
            "type": "line",
            "x0": -5000, 
            "x1": signal_len + 5000, 
            "y0": y_val, 
            "y1": y_val,
            "xref": f"x{row_num}", 
            "yref": f"y{row_num}",
            "line": {"color": color, "width": 3, "dash": "dash"},
            "editable": True,  # Make line draggable
        })
    return shapes

def get_fig_area(data, fig_area_all, shared_triggers, shared_limits):
    """
    Create plot fig
    """
    shapes = []
    row_num = 1
    for name, single_obj in data.items():
        r, r_avg, r_low_lim, r_up_lim, triggers, limits = single_obj.radius_eval()
    
        # Initialize or update shared triggers
        if row_num-1 not in shared_triggers['triggers']:
            shared_triggers['triggers'][row_num-1] = {
                'start': triggers['start'],
                'stop': triggers['stop']
            }
        # Initialize or update shared limits
        if row_num-1 not in shared_limits['limits']:
            shared_limits['limits'][row_num-1] = {
                'alarm_high': limits['alarm_high'],
                'alarm_low': limits['alarm_low'], 
                'warning_high': limits['warning_high'],
                'warning_low': limits['warning_low'],
                'target': limits['target'],
                'freq' : limits['freq'],
                'goal_t': limits['goal_t']
            }
        
        subplot_triggers = shared_triggers['triggers'][row_num-1]
        subplot_limits = shared_limits['limits'][row_num-1]

        shapes.extend(add_trigger_lines(fig_area_all, row_num, 
                                      subplot_triggers, r))
        shapes.extend(add_limit_functions(fig_area_all, 
                                          row_num, len(r_avg), 
                                          subplot_limits))

        area_traces = [
                (r_avg, None, 'Moving average', 'lightblue', None, True),
                ([min(max(val, limits['alarm_low']), limits['alarm_high']) for val in r_avg], 
                 'tonexty', 'Outside of limits', 'rgba(0,0,0,0)', 'rgba(255,0,0,0.5)', True),
                ([min(max(val, limits['warning_low']), limits['alarm_high']) for val in r_avg], 
                 'tonexty', 'Inside radius range', 'rgba(0,0,0,0)', 'rgba(255,165,0,0.3)', False),
                ([max(min(val, limits['warning_high']), limits['alarm_low']) for val in r_avg], 
                 'tonexty', 'Inside radius range', 'rgba(0,0,0,0)', 'rgba(255,165,0,0.5)', True),
                ([min(max(val, limits['target']), limits['warning_high']) for val in r_avg], 
                 'tonexty', 'Optimal radius', 'rgba(0,0,0,0)', 'rgba(11, 156, 49,0.1)', False),
                ([max(min(val, limits['target']), limits['warning_low']) for val in r_avg], 
                 'tonexty', 'Optimal radius', 'rgba(0,0,0,0)', 'rgba(11, 156, 49, 0.3)', True)
            ]
        
        for y, fill, name, line_color, fillcolor, show_legend in area_traces:
            fig_area_all.add_trace(go.Scatter(
                y=y, fill=fill, name=name,
                mode='lines', line_color=line_color,
                fillcolor=fillcolor, legendgroup=row_num,
                showlegend=show_legend
            ), row=row_num, col=1)

        fig_area_all.update_xaxes(
            title_text="Samples", 
            row=row_num, 
            col=1,
            range=[0, len(r)] 
        )
        fig_area_all.update_yaxes(
            title_text="Radius [m]", 
            row=row_num, 
            col=1,
            range=[min(r), max(r)] 
        )
        row_num += 1
        
    fig_area_all.update_layout(
        dragmode=False,  # Disable zoom & pan
        showlegend=True, 
        height=300*row_num, 
        # autosize=True,
        shapes=shapes  # Add shapes to plot
    )
    return fig_area_all

def get_fig_avg(data, fig_mov_avg_all, shared_triggers, shared_limits):
    """
    Create plot fig
    """
    shapes = []
    row_num = 1
    for name, single_obj in data.items():
        r, r_avg, r_low_lim, r_up_lim, triggers, limits = single_obj.radius_eval()
        
        # Initialize or update shared triggers
        if row_num-1 not in shared_triggers['triggers']:
            shared_triggers['triggers'][row_num-1] = {
                'start': triggers['start'],
                'stop': triggers['stop']
            }
            
        # Initialize or update shared limits
        if row_num-1 not in shared_limits['limits']:
            shared_limits['limits'][row_num-1] = {
                'alarm_high': limits['alarm_high'],
                'alarm_low': limits['alarm_low'], 
                'warning_high': limits['warning_high'],
                'warning_low': limits['warning_low'],
                'target': limits['target']
            }

        subplot_triggers = shared_triggers['triggers'][row_num-1]
        subplot_limits = shared_limits['limits'][row_num-1]

        shapes.extend(add_trigger_lines(fig_mov_avg_all, row_num, 
                                      subplot_triggers, r))
        shapes.extend(add_limit_functions(fig_mov_avg_all, 
                                          row_num, len(r_avg),
                                          subplot_limits))

        # Add radius and moving average lines
        traces = [
            (r, 'Radius', 'blue', None),
            (r_avg, 'Moving average', 'lightblue', None)
        ]
        
        for y, name, color, _ in traces:
            fig_mov_avg_all.add_trace(go.Scatter(
                y=y, mode='lines', name=name,
                line=dict(color=color),
                legendgroup=row_num
            ), row=row_num, col=1)

        fig_mov_avg_all.update_xaxes(
            title_text="Samples", 
            row=row_num, 
            col=1,
            range=[0, len(r)] 
        )
        fig_mov_avg_all.update_yaxes(
            title_text="Radius [m]", 
            row=row_num, 
            col=1,
            range=[min(r) - 1, max(r) + 1] 
        )    
        row_num += 1
    
    fig_mov_avg_all.update_layout(
        dragmode=False,  # Disable zoom & pan
        showlegend=True,  
        # autosize=True,
        height=300*row_num,
        shapes=shapes  # Add shapes to plot
    )
    return fig_mov_avg_all

# def get_fig_features(data, fig_feature_all, window_size):
#     """
#     Create plot fig
#     """
#     row_num = 1
#     col_num = 1
#     for name, single_obj in data.items():
#         r, r_avg, r_low_lim, r_up_lim, triggers, limits = single_obj.radius_eval()
#         features_all, colors, names = sliding_window_features(r, window_size)
    
#         for fig_key, fig_features in fig_feature_all.items():
#             for feat_key, y in features_all.items():
#                 if feat_key[-1] == fig_key:
#                     fig_features.add_trace(go.Scatter(
#                         y=y, mode='lines', name=name,
#                         line=dict(color=colors[feat_key]),
#                         legendgroup=str(row_num) + str(col_num)
#                     ), row=row_num, col=col_num)

#                 fig_features.update_xaxes(
#                     title_text="Samples", 
#                     row=row_num, 
#                     col=col_num,
#                     # range=[0, len(r)] 
#                 )
#                 fig_features.update_yaxes(
#                     title_text="Radius [m]", 
#                     row=row_num, 
#                     col=col_num,
#                     # range=[min(r), max(r)] 
#                 )
#             fig_features.update_layout(
#                 dragmode=False,
#                 showlegend=True,
#                 # height=300*row_num, 
#                 autosize=True,
#                 title_text=names[fig_key],
#             )
#         if col_num == 3:
#             col_num = 1
#             row_num += 1
#         else:
#             col_num += 1
#     return fig_feature_all

def get_fig_features(data, fig_feature_all, window_size):
    row_num = 1
    col_num = 1

    for name, single_obj in data.items():
        r, r_avg, r_low_lim, r_up_lim, triggers, limits = single_obj.radius_eval()
        velocity = single_obj.extra_measurements()
        features_all, colors, plot_names = sliding_window_features(r, window_size)
    
        x_options = {
            "samples": np.arange(len(r)),
            "time": np.arange(len(r)) / 100,
            "relative_trigger": velocity
        }

        for fig_key, fig_features in fig_feature_all.items():
            # Plot features that belong to this fig
            for x_label, x_vals in x_options.items():
                for feat_key, y in features_all.items():
                    if feat_key[-1] == fig_key:
                        fig_features.add_trace(
                            go.Scatter(
                                x=x_vals,
                                y=y, 
                                mode='lines', 
                                name= feat_key[:-2],
                                line=dict(color=colors[feat_key]),
                                legendgroup=str(row_num) + str(col_num),
                                showlegend=(row_num == 1 and col_num == 1),  # Only show once to avoid clutter
                                visible=(x_label == "samples") # Only show samples for the first plot
                            ),
                            row=row_num, col=col_num
                        )

            # Only once per subplot: update layout
            fig_features.update_xaxes(
                title_text="Samples", row=row_num, col=col_num
            )
            fig_features.update_yaxes(
                title_text="Radius [m]", row=row_num, col=col_num
            )

        # Advance subplot position
        if col_num == 3:
            col_num = 1
            row_num += 1
        else:
            col_num += 1
    # Update layout for all figures

    for fig_key, fig_features in fig_feature_all.items():
        fig_features.update_layout(
        dragmode=False,
        showlegend=True,
        autosize=True,
        title_text=plot_names.get(fig_key, f"Features {fig_key}"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            title=None
        ),
        annotations=[]  # remove subplot titles
        )
    return fig_feature_all

def update_subplot(df_list, triggers, limits):
    """
    Update subplot with trigger lines and limits
    """
    all_data = {}
    idx = 0
    for tab in df_list:
        datatable = pd.DataFrame(tab['props']['children'][0]['props']['data'])
        file_name = tab['props']['value']
        all_data[file_name] = SK_radius_eval(datatable, triggers['triggers'][str(idx)], limits['limits'][str(idx)])
        idx += 1
    return all_data

def update_figures(data, shared_triggers, shared_limits):
    new_data = update_subplot(data, shared_triggers, shared_limits)
    num_group_subplots = len(new_data)
    file_names = ["Test run " + name[:-4] for name, _ in new_data.items()]
    # Update figures with new data
    fig_area_all, fig_mov_avg_all = create_graph_config(num_group_subplots, file_names)
    area_fig = get_fig_area(new_data, fig_area_all, shared_triggers, shared_limits)
    avg_fig = get_fig_avg(new_data, fig_mov_avg_all, shared_triggers, shared_limits)
    return area_fig, avg_fig

def SK_radius_eval(eval_data, triggers, limits):
    modul_R_param = evlSK.modul_R(eval_data['Radius'], limits)
    R = modul_R_param.compute_module_R()
    module_dH_param = evlSK.modul_dH(eval_data['Lenkradwin'], limits)
    dH = module_dH_param.compute_module_dH()
    modul_t_param = evlSK.modul_t(eval_data['Lenkradwin'], limits)
    t = modul_t_param.compute_module_t()
    evaluation = 0.5 * R + 0.3 * dH + 0.2 * t

    plot_res = evlSK.plotting(modul_R_param, module_dH_param, 1, 
                              triggers, limits, eval_data['Fahrge_DIS'])
    return plot_res

def sliding_window_features(raw, window_size=50):
    n = len(raw)
    max_q = deque()
    min_q = deque()
    
    features = {
        'Maximum_1': [],
        'Minimum_1': [],
        'Standard deviation_2': [],
        'range': [],
        'rms': [],
        'skewness': [],
        'kurtosis': []
    }

    colors = {
        'Maximum_1': 'blue',
        'Minimum_1': 'red',
        'Standard deviation_2': 'green',
        'Range': 'orange',
        'rms': 'purple',
        'Skewness': 'brown',
        'Kurtosis': 'pink'
    }

    names = {
        '1': 'Maximum and Minimum',
        '2': 'Standard Deviation',
    }

    for i in range(n):
        # Deque updates for max
        while max_q and max_q[0] <= i - window_size:
            max_q.popleft()
        while max_q and raw[max_q[-1]] <= raw[i]:
            max_q.pop()
        max_q.append(i)

        # Deque updates for min
        while min_q and min_q[0] <= i - window_size:
            min_q.popleft()
        while min_q and raw[min_q[-1]] >= raw[i]:
            min_q.pop()
        min_q.append(i)

        # Once we have a full window
        if i >= window_size - 1:
            window = raw[i - window_size + 1:i + 1]
            features['Maximum_1'].append(raw[max_q[0]])
            features['Minimum_1'].append(raw[min_q[0]])
            # features['mean'].append(np.mean(window))
            features['Standard deviation_2'].append(np.std(window))
            # features['range'].append(raw[max_q[0]] - raw[min_q[0]])
            # features['rms'].append(np.sqrt(np.mean(window ** 2)))
            # features['skewness'].append(skew(window))
            # features['kurtosis'].append(kurtosis(window))

    return features, colors, names