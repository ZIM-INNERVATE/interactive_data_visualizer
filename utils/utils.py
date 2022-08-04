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

matplotlib.use('Agg')

def save_file(name, content, upload_directory):
    """
    Encode data to base64 and save it to the specified upload directory
    """
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(upload_directory, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

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

def get_dataframe(filename, upload_directory, delimeter=","):
    """
    Read csv file
    """
    return pd.read_csv(os.path.join(upload_directory, filename),
                       delimiter=delimeter).reset_index()

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
    pca = PCA(n_components=len(data), whiten=False)
    
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

def get_fig(data, fig_pca_all, fig_pca_group, fig_qq_plot):
    """
    Create plot fig for pca all, pca group, and qq plot
    """
    x_length = np.linspace(-2,2,100)
    bincount = 33
    row_num = 1

    for i,motion in enumerate(data.keys()):
        for j,group in enumerate(data[motion].keys()):
            if group == "all":
                for k,dist in enumerate(["px", "py", "theta"]):
                    dist_data = data[motion]["all"][dist]
                    dist_data = np.asarray(dist_data)
                    if dist != "theta":
                        # create pca all plot
                        fig_pca_all.append_trace(go.Scatter(
                            x=x_length,
                            y=stats.norm.pdf(x_length, loc=dist_data.mean(), scale=dist_data.std()),
                            name=f"{motion}: PCA-all {dist} dist",
                            legendgroup = i
                            ), 
                            row=i+1, col=1
                        )
                        fig_pca_all.append_trace(go.Histogram(
                            x=dist_data,
                            nbinsy=bincount,
                            opacity=0.5,
                            histnorm='probability density',
                            name=f"{motion}: PCA-all {dist} hist",
                            legendgroup = i
                            ),  
                            row=i+1, col=1
                        )
                    # create qq plot
                    qqplot_data = qqplot(dist_data, 
                                         loc=dist_data.mean(),
                                         scale=dist_data.std(),
                                         line='45')
                                         
                    mpl_lines = qqplot_data.gca().lines
                    fig_qq_plot.append_trace(go.Scatter(
                        x=mpl_lines[0].get_xdata(),
                        y= mpl_lines[0].get_ydata(),
                        mode='markers',
                        marker= {'color': 'blue'}

                        ), 
                        row=i+1, col=k+1
                    )
                    fig_qq_plot.append_trace(go.Scatter(
                        x=mpl_lines[1].get_xdata(),
                        y=mpl_lines[1].get_ydata(),
                        mode='lines',
                        marker= {'color': 'red'}

                        ), 
                        row=i+1, col=k+1
                    )
                    
                    # close figure to avoid OOM
                    matplotlib.pyplot.close(qqplot_data)
            else:
                for k,dist in enumerate(["px", "py", "theta"]):
                    if dist != "theta":
                        dist_data = data[motion][group][dist]
                        dist_data = np.asarray(dist_data)
                        fig_pca_group.append_trace(go.Scatter(
                            x=x_length,
                            y=stats.norm.pdf(x_length, loc=dist_data.mean(), scale=dist_data.std()),
                            name=f"{motion}-{group}: {dist} distribution",
                            legendgroup = row_num
                            ), 
                            row=row_num, col=1
                        )
                        fig_pca_group.append_trace(go.Histogram(
                            x=dist_data,
                            nbinsy=bincount,
                            opacity=0.5,
                            histnorm='probability density',
                            name=f"{motion}-{group}: {dist} histogram",
                            legendgroup = row_num
                            ),  
                            row=row_num, col=1
                        )

                row_num += 1

    return fig_pca_all, fig_pca_group, fig_qq_plot

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
