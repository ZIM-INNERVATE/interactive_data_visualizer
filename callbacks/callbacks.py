import os

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils.utils as utils
from app import app
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots

UPLOAD_DIRECTORY = "/tmp"
if os.environ.get("DATADIR") is not None:
    UPLOAD_DIRECTORY = os.environ.get("DATADIR")

@app.callback([Output("file-list", "children"),
               Output("dropdown-select-file", "options"),],
              [Input("upload-data", "filename"),
               Input("upload-data", "contents"),],
             )
def file_list(uploaded_filenames, 
              uploaded_file_contents):
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        utils.save_file(uploaded_filenames, uploaded_file_contents, UPLOAD_DIRECTORY)

    files = utils.get_uploaded_files(UPLOAD_DIRECTORY)
    if len(files) == 0:
        return [dash.html.Li("No files uploaded")], []
    else:
        return [dash.html.Li(fname) for fname in files],\
               [{'label': fname, 'value': fname} for fname in  files]

@app.callback(Output('output-selected-file', 'children'),
              [Input('dropdown-select-file', 'value'),
               Input('select-button', 'n_clicks'),
               Input('delete-button', 'n_clicks')],
             )
def process_selected_file(selected_file, n_clicks_select, n_clicks_delete):
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'delete-button' and selected_file is not None:
            utils.delete_file(selected_file, UPLOAD_DIRECTORY)
        elif button_id == 'select-button' and selected_file is not None:
            return selected_file
    else:
        return None
        
@app.callback([Output('checklist-motions', 'value'),
               Output('checklist-motions-all', 'value')],
              [Input('checklist-motions-all', 'value'),
               Input('checklist-motions', 'value'),
               State('checklist-motions', 'options')],
              prevent_initial_call=True,
             )
def update_motion_checklist(checklist_motions_all_val,
                            checklist_motions_val, 
                            checklist_motions_opt):
    ctx = dash.callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if input_id == "checklist-motions-all":
        if checklist_motions_all_val:
            checklist_motions_val = [val['value'] for val in checklist_motions_opt]
        else:
            checklist_motions_val = []
        return checklist_motions_val, checklist_motions_all_val
    elif input_id == "checklist-motions":
        if len(checklist_motions_opt) == len(checklist_motions_val):
            return checklist_motions_val, ["All"]
        else:
            return checklist_motions_val, []

@app.callback([Output('checklist-weights', 'value'),
               Output('checklist-weights-all', 'value')],
              [Input('checklist-weights-all', 'value'),
               Input('checklist-weights', 'value'),
               State('checklist-weights', 'options')],
              prevent_initial_call=True,
             )
def update_weight_checklist(checklist_weights_all_val,
                            checklist_weights_val, 
                            checklist_weights_opt):
    ctx = dash.callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if input_id == "checklist-weights-all":
        if checklist_weights_all_val:
            checklist_weights_val = [val['value'] for val in checklist_weights_opt]
        else:
            checklist_weights_val = []
        return checklist_weights_val, checklist_weights_all_val
    elif input_id == "checklist-weights":
        if len(checklist_weights_opt) == len(checklist_weights_val):
            return checklist_weights_val, ["All"]
        else:
            return checklist_weights_val, []

@app.callback([Output('xy-scatter', 'figure'),
               Output('pca-all-fig', 'figure'),
               Output('pca-group-fig', 'figure'),
               Output('normality-test-fig', 'figure')],
              [Input('experimentation-data', 'data'),],
              prevent_initial_call=True,
             )
def update_graphs(experimentation_data):
    if experimentation_data:
        df = pd.DataFrame.from_records(experimentation_data)
        num_selected_weights = 1
        if 'weight' in df:
            num_selected_weights = len(df.weight.unique())
        num_group_subplots = len(df.group.unique()) * \
                              len(df.motion.unique()) * \
                              num_selected_weights
        if not df.empty:
            extracted_df = utils.exctract_data(df)
            # create scatter plot
            fig_scatter = px.scatter(df,
                        x="x",
                        y="y",
                        title="End robot position for left, straight, and right motions",
                        color=df["group"].astype(str),
                        hover_data=["group"],
                        template='plotly',
                        height=800,
                        width=1200,
            )
            fig_scatter.update_layout(legend_title="Group")
            fig_scatter.update_layout(title_x=0.5)

            # Create distribution and hist plot for all data
            fig_pca_all = make_subplots(rows=3, cols=1, 
                                    subplot_titles=(" "," "," "), 
                                    #shared_xaxes=True,
                                    )
            fig_pca_group = make_subplots(rows=num_group_subplots, cols=1, 
                                    subplot_titles=tuple([" "]*num_group_subplots), 
                                    #shared_xaxes=True,
                                    )
            fig_qq_plot = make_subplots(rows=3, cols=3, 
                                        subplot_titles=tuple([" "]*9), 
                                        #shared_xaxes=True,
                                        horizontal_spacing = 0.05)
            # update layout for pca all
            fig_pca_all.update_layout(title_text="2D pose distribution and histogram after PCA projection on all data",
                                  #title_x=0.5, #center title
                                  height=800,
                                  width=1200,
                                  autosize = True,
                                  legend_tracegroupgap = 180,)
            # update layout for pca group
            fig_pca_group.update_layout(title_text="2D pose distribution and histogram after PCA projection on each group",
                      #title_x=0.5, #center title
                      height=800*num_group_subplots/3,
                      width=1200,
                      autosize = True,
                      legend_tracegroupgap = 190,)
            # update layout on qq plot on all data
            fig_qq_plot.update_layout(title_text="Quantile-Quantile Plot (QQ Plot) on all data",
                      height=1200,
                      width=1200,
                      autosize = True,
                      showlegend=False,)
            
            # get pca all, pca group, qq plot
            fig_pca_all, fig_pca_group, fig_qq_plot = utils.get_fig(extracted_df, 
                                                                    fig_pca_all, 
                                                                    fig_pca_group, 
                                                                    fig_qq_plot)

            # update subplot title and axes label
            pca_plt_num = 0
            qq_plt_num = 0
            for i,motion in enumerate(extracted_df):
                for j,group in enumerate(extracted_df[motion]):
                    if group == "all":
                        for k,dist in enumerate(["px", "py", "theta"]):
                            # qq subplot title
                            fig_qq_plot.layout.annotations[qq_plt_num].update(text=f"{motion} motion - {dist}")
                            fig_qq_plot.update_xaxes(
                                        title_text= "Theoritical Quantities",
                                        zeroline= False,
                                        row=i+1, col=k+1
                            )
                            fig_qq_plot.update_yaxes(
                                        title_text= "Sample Quantities",
                                        row=i+1, col=k+1
                            )
                            qq_plt_num += 1
                    else:
                        # group pca subplot title
                        fig_pca_group.layout.annotations[pca_plt_num].update(text=f"{motion} motion - group {group} pose distribution")
                        pca_plt_num += 1
                
                # all pca subplot title
                fig_pca_all.layout.annotations[i].update(text=f"{motion} motion pose distribution")

            return fig_scatter, fig_pca_all, fig_pca_group, fig_qq_plot
        else:
            return px.scatter(), px.scatter(), px.scatter(), px.scatter()
    else:
        raise dash.exceptions.PreventUpdate

@app.callback([Output('normality-test-table-lilliefors', 'data'),
               Output('normality-test-table-lilliefors', 'columns'),
               Output('normality-test-table-shapiro', 'data'),
               Output('normality-test-table-shapiro', 'columns'),
               Output('normality-test-table-chi2', 'data'),
               Output('normality-test-table-chi2', 'columns')],
              [Input('experimentation-data', 'data'),],
              prevent_initial_call=True,
             )
def update_normality_table(experimentation_data):
    if experimentation_data:
        significance = 0.05
        df = pd.DataFrame.from_records(experimentation_data)  
        if not df.empty:
            extracted_df = utils.exctract_data(df)
            test_result_all_lilliefors = []
            test_result_all_shapiro = []
            test_result_all_chi2 = []
            test_result_all_header = []
            for i,motion in enumerate(extracted_df):
                for j,group in enumerate(extracted_df[motion]):
                    if group == "all":
                        for k,dist in enumerate(["px", "py", "theta"]):
                            dist_data = extracted_df[motion]["all"][dist]
                            dist_data = np.asarray(dist_data)
                            
                            l_ksstat, l_pval = utils.compute_lilliefors(dist_data)
                            s_stat, s_pval = utils.compute_shapiro(dist_data)
                            chi2_pval = utils.compute_chi2(dist_data, significance)

                            test_result_all_lilliefors.append({"Motion": f"{motion}-{dist}",
                                                    "Group": group,
                                                    "Significnce": significance,
                                                    "Stat": round(l_ksstat,4),
                                                    "P Value": round(l_pval, 4),
                                                    "Reject H0": "Yes" if l_pval < significance else "No",
                                                    })

                            test_result_all_shapiro.append({"Motion": f"{motion}-{dist}",
                                                    "Group": group,
                                                    "Significnce": significance,
                                                    "Stat": round(s_stat,4),
                                                    "P Value": round(s_pval, 4),
                                                    "Reject H0": "Yes" if s_pval < significance else "No",
                                                    })
                            test_result_all_chi2.append({"Motion": f"{motion}-{dist}",
                                                    "Group": group,
                                                    "Significnce": significance,
                                                    "Stat": 0,
                                                    "P Value": round(chi2_pval, 4),
                                                    "Reject H0": "Yes" if chi2_pval < significance else "No",
                                                    })

            test_result_all_header = [{"name": i, "id": i} for i in test_result_all_lilliefors[0].keys()]
            return test_result_all_lilliefors, test_result_all_header, \
                    test_result_all_shapiro, test_result_all_header, \
                    test_result_all_chi2, test_result_all_header
        else:
            return None, None, None, None, None, None
    else:
        raise dash.exceptions.PreventUpdate

@app.callback([Output('experimentation-data', 'data'),
               Output('experimentation-data', 'columns'),
               Output('dropdown-groups', 'options'),
               Output('dropdown-groups', 'value'),
               Output('checklist-motions', 'options'),
               Output('checklist-weights', 'options'),],
              [Input('output-selected-file', 'children'),
               Input('dropdown-csv-separator', 'value'),
               Input('dropdown-groups', 'value'),
               Input('checklist-motions', 'value'),
               Input('checklist-weights', 'value')],
              prevent_initial_call=False,
             )
def update_data_table(selected_file,
                      csv_separator,
                      selected_groups,
                      selected_motions,
                      selected_weights):
    if selected_file:
        df = utils.load_dataframe(os.path.join(UPLOAD_DIRECTORY, selected_file),
                                 csv_separator)
        if not df.empty:
            # generate motion, weight, and group checklist
            checklist_motions = [{"label": col, "value": col} for col in df.motion.unique()]
            if "weight" in df:
                checklist_weights = [{"label": col, "value": col} for col in df.weight.unique()]
            else:
                checklist_weights = [{"label": "small (NA)", "value": "small"},
                                    {"label": "medium (NA)", "value": "medium"},
                                    {"label": "large (NA)", "value": "large"}]
            group_ids = df.group.unique()

            df = df[df["group"].isin(selected_groups)]
            df = df[df["motion"].isin(selected_motions)]
            if 'weight' in df:
                df = df[df["weight"].isin(selected_weights)]
            df.reset_index(drop=True, inplace=True)

            header = [{"name": i, "id": i} for i in df.columns]
            data = df.to_dict('records')
            return data, \
                   header, \
                   group_ids,\
                   group_ids,\
                   checklist_motions,\
                   checklist_weights
        else:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate
