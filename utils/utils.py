import base64
import os
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smstatsd
from scipy.stats import shapiro
from sklearn.decomposition import PCA
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import lilliefors

import h5py
import importlib
import utils.SK_evaluation as evlSK
importlib.reload(evlSK)

matplotlib.use('Agg')

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

def load_dataframe(filepath, sep=","):
    """
    Read csv file
    """
    df = pd.read_csv(filepath,
                     sep=sep,
                     dtype="str").reset_index()
    # convert x, y, and theta to float if separator is ;, the decimal point os ,
    df['x'] = df['x'].apply(lambda x: float(x.split()[0].replace(',', '.')))
    df['x'] = df['x'].astype(float)

    df['y'] = df['y'].apply(lambda x: float(x.split()[0].replace(',', '.')))
    df['y'] = df['y'].astype(float)

    df['theta'] = df['theta'].apply(lambda x: float(x.split()[0].replace(',', '.')))
    df['theta'] = df['theta'].astype(float)

    return df

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
    df = pd.DataFrame(data=np.array([[start, stop]]), columns=['Start Trigger', 'Stop Trigger'])
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

def exctract_data(df):
    """
    Extract dataframe and store them in a dictionary
    """
    # add the data to a dictionary
    obs_data = defaultdict(dict)
    for motion in df.motion.unique():
        obs_data[motion]["all"] = {}
        obs_data[motion]["all"]["x"] = []
        obs_data[motion]["all"]["y"] = []
        obs_data[motion]["all"]["theta"] = []
        obs_data[motion]["all"]["px"] = []
        obs_data[motion]["all"]["py"] = []
        for group in df.group.unique():
            obs_data[motion][group] = {}
            obs_data[motion][group]["x"] = df.loc[df["motion"] == motion].loc[df["group"] == group]["x"]
            obs_data[motion][group]["y"] = df.loc[df["motion"] == motion].loc[df["group"] == group]["y"]
            obs_data[motion][group]["theta"] = df.loc[df["motion"] == motion].loc[df["group"] == group]["theta"]
            
            transformed_xy = transform_data([obs_data[motion][group]["x"], obs_data[motion][group]["y"]])
            obs_data[motion][group]["px"] = transformed_xy[:,0]
            obs_data[motion][group]["py"] = transformed_xy[:,1]
            obs_data[motion]["all"]["x"].extend(obs_data[motion][group]["x"])
            obs_data[motion]["all"]["y"].extend(obs_data[motion][group]["y"])
            obs_data[motion]["all"]["theta"].extend(obs_data[motion][group]["theta"])
            
        transformed_all_xy = transform_data([obs_data[motion]["all"]["x"], obs_data[motion]["all"]["y"]])
        obs_data[motion]["all"]["px"] = transformed_all_xy[:,0]
        obs_data[motion]["all"]["py"] = transformed_all_xy[:,1]
            
    return obs_data

def extract_data(df_list, triggers):
    """
    Extract dataframe and store them in a dictionary
    """

    all_data = {}
    for tab in df_list:
        datatable = pd.DataFrame(tab['props']['children'][0]['props']['data'])
        file_name = tab['props']['value']
        evaluation_list = SK_radius_eval(datatable, triggers)
        all_data[file_name] = evaluation_list
    return all_data

def combine_tails(hist, bin_edges, expected_freq=5, tail=0):
    """
    Combine tails if number of bin count is less than expected freq
    """
    if hist[tail] < expected_freq:
        #todo: does not work with [5,1,1,8,11]
        count = hist.pop(tail)
        hist[tail] += count
        
        # pop the next edge, and keep leftmost/rightmost edge
        if tail == 0: bin_edges.pop(tail+1)
        else: bin_edges.pop(tail-1)
        
        return combine_tails(hist, bin_edges, expected_freq, tail)
    else:
        return hist, bin_edges

def combine_bins(hist, bin_edges, expected_freq=5):
    """
    Combine bins in both tails if it's less than expected frequency.
    For a chi-square test to be valid, the expected frequency should be at least 5 
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm
    """
    if len(hist) <= 1:
        return hist, bin_edges
    
    # combine lower tail
    hist, bin_edges = combine_tails(hist, bin_edges, expected_freq, 0)
    
    # combile upper tail
    hist, bin_edges = combine_tails(hist, bin_edges, expected_freq, -1)
    
    return hist, bin_edges

def get_fig(data, fig_mov_avg_all, fig_area_all):
    """
    Create plot fig for pca all, pca group, and qq plot
    """
    x_length = np.linspace(-2,2,100)
    bincount = 33

    row_num = 1
    for name, single_obj in data.items():
        r, r_avg, r_low_lim, r_up_lim, triggers = single_obj.radius_eval()
        fig_mov_avg_all.add_trace(go.Scatter(
            y=r,
            mode='lines',
            name='Radius',
            line=dict(color='blue'),
            legendgroup = str(row_num)
        ), row=row_num, col=1)

        # Add moving average line
        fig_mov_avg_all.add_trace(go.Scatter(
            y=r_avg,
            mode='lines',
            name='Moving average',
            line=dict(color='lightblue'),
            legendgroup = str(row_num)
        ), row=row_num, col=1)

        # Add area under average line
        fig_area_all.add_trace(
            go.Scatter(
            y=r_avg,
            mode='lines',
            name='Moving average',
            line=dict(color='lightblue'),
            legendgroup = str(row_num)
        ), row=row_num, col=1)

        # Add horizontal lines - common traces
        trace1 = go.Scatter(
            y=[r_low_lim] * len(r),
            mode='lines',
            name='Alarm',
            line=dict(color='red', dash='dash'),
            legendgroup = str(row_num)
        )
        trace2 = go.Scatter(
            y=[r_up_lim] * len(r),
            mode='lines',
            showlegend=False,
            line=dict(color='red', dash='dash'),
            legendgroup = str(row_num)
        )
        trace3 = go.Scatter(
            y=[39] * len(r),
            mode='lines',
            name='Warning',
            line=dict(color='green', dash='dash'),
            legendgroup = str(row_num)
        )
        trace4 = go.Scatter(
            y=[40] * len(r),
            mode='lines',
            name='Target',
            line=dict(color='black', dash='dash'),
            legendgroup = str(row_num)
        )
        trace5 = go.Scatter(
            y=[41] * len(r),
            mode='lines',
            showlegend=False,
            line=dict(color='green', dash='dash'),
            legendgroup = str(row_num)
        )
        fig_mov_avg_all.add_trace(trace1, row=row_num, col=1)
        fig_mov_avg_all.add_trace(trace2, row=row_num, col=1)
        fig_mov_avg_all.add_trace(trace3, row=row_num, col=1)
        fig_mov_avg_all.add_trace(trace4, row=row_num, col=1)
        fig_mov_avg_all.add_trace(trace5, row=row_num, col=1)

        fig_area_all.add_trace(trace1, row=row_num, col=1)
        fig_area_all.add_trace(trace2, row=row_num, col=1)
        fig_area_all.add_trace(trace3, row=row_num, col=1)
        fig_area_all.add_trace(trace4, row=row_num, col=1)
        fig_area_all.add_trace(trace5, row=row_num, col=1)
        
        # Add vertical lines for triggers
        
        fig_mov_avg_all.add_vline(
            x=triggers['Start Trigger'],
            line_dash="dashdot",
            line_color="green",
            name="Start trigger",
            legendgroup = str(row_num),
            row=row_num, col=1
        )
        
        fig_mov_avg_all.add_vline(
            x=triggers['Stop Trigger'],
            line_dash="dashdot",
            line_color="red",
            name="Stop trigger",
            legendgroup = str(row_num),
            row=row_num, col=1
        )

        fig_mov_avg_all.update_xaxes(
            title_text= "Samples",
            row=row_num, col=1
        )

        fig_mov_avg_all.update_yaxes(
            title_text="Radius [m]", 
            row=row_num, col=1
        ) 

        fig_area_all.add_vline(
            x=triggers['Start Trigger'],
            line_dash="dashdot",
            line_color="green",
            name="Start trigger",
            legendgroup = str(row_num),
            row=row_num, col=1
        )
        
        fig_area_all.add_vline(
            x=triggers['Stop Trigger'],
            line_dash="dashdot",
            line_color="red",
            name="Stop trigger",
            legendgroup = str(row_num),
            row=row_num, col=1
        )

        fig_area_all.update_xaxes(
            title_text= "Samples",
            row=row_num, col=1
        )

        fig_area_all.update_yaxes(
            title_text="Radius [m]", 
            row=row_num, col=1
        ) 

        row_num += 1
    return fig_mov_avg_all, fig_area_all

def compute_lilliefors(data):
    """
    Compute stat and pval with lilliefors
    """
    ksstat, pvalue = lilliefors(data, pvalmethod="table")

    return ksstat, pvalue

def compute_chi2(data, significance, hist_bins=11, expected_freq=5):
    """
    Compute pval with chi-square
    """
    hist, bin_edges = np.histogram(data, bins=hist_bins)
    hist = hist.tolist()
    bin_edges = bin_edges.tolist()
    hist, bin_edges  = combine_bins(hist, bin_edges, expected_freq=expected_freq)

    # Calculate Expected value: discrete difference * sum(hist)
    f_exp = np.diff(stats.norm.cdf(bin_edges, loc=data.mean(), scale=data.std()))*sum(hist)
    
    # chi2sum = (B-E)^2/E
    # B=hist: Observed (Häufigkeiten), f=exp: Expected (Erwartete (angepasste) Häufigkeit)
    chi2sum = sum((np.array(hist)-f_exp)**2/f_exp)
    
    # there was a mistake in PGP sample, x^2 = (B-E)^2/E
    #chi2sum = sum((f_exp-np.array(hist))**2/f_exp)
            
    #how to determine degree of freedom???
    #dof is determine by the number of histogram - number of parameter being estimated (mean, std)
    #d=k-1-2 (2 for mean and std)
    dof = len(hist)-3
    
    # 95% quantile
    quantile_95 = stats.chi2.ppf(1-significance,df=dof)
        
    # fixme: sometimes returns nan
    chi2_result = 1 - stats.chi2.cdf(chi2sum,df=dof)

    return chi2_result

def compute_shapiro(data):
    """
    Evaluate data using Shapiro-Wilk test, and determine whether the data was drawn 
    from Gaussian distribution
    """
    shapiro_stat, shapiro_pval = shapiro(data)
    return shapiro_stat, shapiro_pval

def SK_radius_eval(eval_data, triggers):
    
    modul_R_param = evlSK.modul_R(eval_data['Radius'])
    R = modul_R_param.compute_module_R()
    module_dH_param = evlSK.modul_dH(eval_data['Lenkradwin'])
    dH = module_dH_param.compute_module_dH()
    modul_t_param = evlSK.modul_t(eval_data['Lenkradwin'])
    t = modul_t_param.compute_module_t()
    evaluation = 0.5 * R + 0.3 * dH + 0.2 * t

    plot_res = evlSK.plotting(modul_R_param, module_dH_param, 1, triggers)
    return plot_res