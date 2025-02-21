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
        valid_files = [fname for fname in files if os.path.splitext(fname)[1] in [".csv", ".hdf5"]]
        return [dash.html.Li(fname) for fname in valid_files],\
               [{'label': fname, 'value': fname} for fname in  valid_files]

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

@app.callback([Output('mov_avg_radius', 'figure'),
               Output("store_avg", "data"),],
              [Input('hdf5-data-tabs', 'children'),
               Input('triggers-data', 'data'),
               Input('mov_avg_radius', 'clickData'),
               Input('mov_avg_radius', 'hoverData'),
               Input('mov_avg_radius', 'relayoutData'),],
              [State("store_avg", "data"),
               State('mov_avg_radius', 'figure'),],
            #   prevent_initial_call=True,
             )
def update_graphs_avg(hdf5_experimentation_data, triggers_data,
                  clickData, hoverData, relayoutData, store_data, figure):
    if hdf5_experimentation_data:
        df_list = hdf5_experimentation_data # pd.DataFrame.from_records(hdf5_experimentation_data)
        if df_list and figure == None:
            num_group_subplots = len(df_list)
            triggers_data = triggers_data[0]
            extracted_df = utils.extract_data(df_list, triggers_data)
            file_names = ["Test run " + name[:-4] for name, _ in extracted_df.items()]

            fig_mov_avg_all = make_subplots(rows=num_group_subplots, cols=1, 
                                    subplot_titles=tuple(file_names), 
                                    vertical_spacing=0.045,
                                    )
            fig_mov_avg_all.update_layout(title_text="<b>Moving average for all runs in the group<b>",
                    title_x=0.5,
                    dragmode=False,
                    height=900*num_group_subplots/3.,
                    showlegend=True,
                    legend_tracegroupgap = 80*num_group_subplots/3.25,
            )
            fig_mov_avg_all = utils.get_fig_avg(extracted_df, fig_mov_avg_all)
            return fig_mov_avg_all, store_data
        
        elif figure != None: 
            fig_mov_avg_all = go.Figure(figure)
            vertical_threshold = 100
            if clickData:
                point = clickData["points"][0]
                x = point["x"]
                # Sensitivity for detecting line clicks
                if abs(x - store_data["x_start_trigger"]) < vertical_threshold:
                    store_data["dragging"] = "start_trigger"  # Start dragging vertical line
                elif abs(x - store_data["x_stop_trigger"]) < vertical_threshold:
                    store_data["dragging"] = "stop_trigger"  # Start dragging vertical line
            if hoverData and store_data["dragging"]:
                point = hoverData["points"][0]
                x = point["x"]
                print('X Position:', x)
                # Print when hovering over lines
                if abs(x - store_data["x_start_trigger"]) < vertical_threshold:
                    print(f"{x}: start trigger")
                if abs(x - store_data["x_stop_trigger"]) < vertical_threshold:
                    print(f"{x}: stop trigger")
                shapes = fig_mov_avg_all["layout"]["shapes"]
                
                # Smoothly move the vertical line
                if store_data["dragging"] == "start_trigger":
                    shapes[0]["x0"] = shapes[0]["x1"] = x  # Move vertical line
                    store_data["x_start_trigger"] = x  # Update the x-coordinate of the line
                elif store_data["dragging"] == "stop_trigger":
                    shapes[1]["x0"] = shapes[1]["x1"] = x  # Move vertical line 2
                    store_data["x_stop_trigger"] = x  
                fig_mov_avg_all["layout"]["shapes"] = shapes  # Apply changes to the figure
            
            if relayoutData:
                store_data["dragging"] = None  # Stop dragging when the mouse is released
            return fig_mov_avg_all, store_data  #fig_scatter (as first argument) fig_qq_plot(as last argument)
        else:
            return px.scatter(), None
    else:
        raise dash.exceptions.PreventUpdate
    
@app.callback([Output('area_under_radius', 'figure'),
               Output("store_area", "data"),],
              [Input('hdf5-data-tabs', 'children'),
               Input('triggers-data', 'data'),
               Input('area_under_radius', 'clickData'),
               Input('area_under_radius', 'hoverData'),
               Input('area_under_radius', 'relayoutData'),],
              [State("store_area", "data"),
               State('area_under_radius', 'figure'),],
            #   prevent_initial_call=True,
             )
def update_graphs_area(hdf5_experimentation_data, triggers_data,
                  clickData, hoverData, relayoutData, store_data, figure):
    if hdf5_experimentation_data:
        df_list = hdf5_experimentation_data 
        if df_list:
            num_group_subplots = len(df_list)
            triggers_data = triggers_data[0]
            extracted_df = utils.extract_data(df_list, triggers_data)
            file_names = ["Test run " + name[:-4] for name, _ in extracted_df.items()]
            fig_area_all = make_subplots(rows=num_group_subplots, cols=1, 
                                    subplot_titles=tuple(file_names), # tuple([" "]*num_group_subplots)
                                    vertical_spacing=0.045,
                                    )
            fig_area_all.update_layout(title_text="<b>Area under the radius curve for all runs in the group<b>",
                    title_x=0.5,
                    dragmode=False,
                    height=900*num_group_subplots/3.,
                    # autosize=True,
                    showlegend=True,
                    legend_tracegroupgap = 80*num_group_subplots/4.07,
                )
            fig_area_all = utils.get_fig_area(extracted_df, fig_area_all) 
            return fig_area_all, store_data  
        else:
            return px.scatter(), None
    else:
        raise dash.exceptions.PreventUpdate

@app.callback([Output('normality-test-table-lilliefors', 'data'),
               Output('normality-test-table-lilliefors', 'columns'),
               Output('normality-test-table-shapiro', 'data'),
               Output('normality-test-table-shapiro', 'columns'),
               Output('normality-test-table-chi2', 'data'),
               Output('normality-test-table-chi2', 'columns')],
              [Input('hdf5-data-tabs', 'children'),
               Input('triggers-data', 'data'),
               Input('fake', 'data')],
              prevent_initial_call=True,
             )
def update_normality_table(hdf5_experimentation_data, triggers, fake):
    if hdf5_experimentation_data:
        significance = 0.05
        # df = pd.DataFrame.from_records(hdf5_experimentation_data)
        df = hdf5_experimentation_data
        if df:
            extracted_df = utils.extract_data(df, triggers) #utils.exctract_data(df)
            # test_result_all_lilliefors = []
            # test_result_all_shapiro = []
            # test_result_all_chi2 = []
            # test_result_all_header = []
            # for i,motion in enumerate(extracted_df):
                # for j,group in enumerate(extracted_df[motion]):
                    # if group == "all":
                        # for k,dist in enumerate(["px", "py", "theta"]):
                            # dist_data = extracted_df[motion]["all"][dist]
                            # dist_data = np.asarray(dist_data)
                            # 
                            # l_ksstat, l_pval = utils.compute_lilliefors(dist_data)
                            # s_stat, s_pval = utils.compute_shapiro(dist_data)
                            # chi2_pval = utils.compute_chi2(dist_data, significance)
# 
                            # test_result_all_lilliefors.append({"Motion": f"{motion}-{dist}",
                                                    # "Group": group,
                                                    # "Significnce": significance,
                                                    # "Stat": round(l_ksstat,4),
                                                    # "P Value": round(l_pval, 4),
                                                    # "Reject H0": "Yes" if l_pval < significance else "No",
                                                    # })
# 
                            # test_result_all_shapiro.append({"Motion": f"{motion}-{dist}",
                                                    # "Group": group,
                                                    # "Significnce": significance,
                                                    # "Stat": round(s_stat,4),
                                                    # "P Value": round(s_pval, 4),
                                                    # "Reject H0": "Yes" if s_pval < significance else "No",
                                                    # })
                            # test_result_all_chi2.append({"Motion": f"{motion}-{dist}",
                                                    # "Group": group,
                                                    # "Significnce": significance,
                                                    # "Stat": 0,
                                                    # "P Value": round(chi2_pval, 4),
                                                    # "Reject H0": "Yes" if chi2_pval < significance else "No",
                                                    # })
# 
            # test_result_all_header = [{"name": i, "id": i} for i in test_result_all_lilliefors[0].keys()]
            # return test_result_all_lilliefors, test_result_all_header, \
                    # test_result_all_shapiro, test_result_all_header, \
                    # test_result_all_chi2, test_result_all_header
        else:
            return None, None, None, None, None, None
    else:
        raise dash.exceptions.PreventUpdate

@app.callback([Output('metadata-data', 'data'),
               Output('metadata-data', 'columns'),
               Output('triggers-data', 'data'),
               Output('triggers-data', 'columns'),],
              [Input('output-selected-file', 'children'),],
              prevent_initial_call=False,
             )
def update_data_table(selected_file):
    if selected_file:
        df_metadata = utils.load_metadata(os.path.join(UPLOAD_DIRECTORY, selected_file))
        df_triggers = utils.load_triggers(os.path.join(UPLOAD_DIRECTORY, selected_file))
        if not df_metadata.empty and not df_triggers.empty:
            df_metadata.reset_index(drop=True, inplace=True)
            header_metadata = [{"name": i, "id": i} for i in df_metadata.columns]
            table_metadata = df_metadata.to_dict('records')

            df_triggers.reset_index(drop=True, inplace=True)
            header_triggers = [{"name": i, "id": i} for i in df_triggers.columns]
            table_triggers = df_triggers.to_dict('records')
            return table_metadata, header_metadata, table_triggers, header_triggers
        else:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate

@app.callback([Output('dropdown-groups', 'options'),
               Output('dropdown-groups', 'value'),
            #Output('dropdown-groups', 'options'),
            #Output('dropdown-groups', 'value'),
            #Output('checklist-motions', 'options'),
            #Output('checklist-weights', 'options'),
               ],
              [Input('output-selected-file', 'children'),],
             )
def initialize_options(selected_file):
    if selected_file:
        df_list = utils.load_dataframe_hdf5(os.path.join(UPLOAD_DIRECTORY, selected_file)) #load_dataframe
        # checklist_motions = [{"label": col, "value": col} for col in df.motion.unique()]
    
        # if "weight" in df:
            # checklist_weights = [{"label": col, "value": col} for col in df.weight.unique()]
        # else:
            # checklist_weights = [{"label": "small (NA)", "value": "small"},
                                #  {"label": "medium (NA)", "value": "medium"},
                                #  {"label": "large (NA)", "value": "large"}]
        return df_list,\
               df_list.keys()
            # df.group.unique(),\
            #   df.group.unique(),\
            #   checklist_motions ,\
            #    checklist_weights
    else:
        raise dash.exceptions.PreventUpdate

@app.callback(
    [Output('hdf5-data-tabs', 'children'),
     Output('hdf5-data-tabs', 'value')],
    [Input('output-selected-file', 'children')],
    prevent_initial_call=True
)
def initialize_options_hdf5(selected_file):
    from dash import dash_table, dcc, html

    if selected_file and selected_file.endswith('.hdf5'):
        data_arrays = utils.load_dataframe_hdf5(os.path.join(UPLOAD_DIRECTORY, selected_file))
        intex_tab = 1
        tabs = []
        for table_name, df in data_arrays.items():
            tabs.append(
                dcc.Tab(
                    label=f'Table-{intex_tab}', #label=table_name[:-4],
                    value=f'{table_name}',
                    children=[
                        dash_table.DataTable(
                            id=f'{table_name}',
                            data=df.to_dict('records'),
                            columns=[{'name': col, 'id': col} for col in df.columns],
                            style_data={'color': 'black'},
                            style_cell={'textAlign': 'center'},
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(220, 220, 220)',
                                    'textAlign': 'center'
                                }
                            ],
                            page_size=25,
                            style_table={'height': '720px', 'overflowY': 'auto'},
                            style_header={
                                'backgroundColor': 'rgb(210, 210, 210)',
                                'color': 'black',
                                'fontWeight': 'bold',
                                'textAlign': 'center'
                            }
                        ),
                    ],
                    className='custom-tab',
                    selected_className='custom-tab--selected',
                )
            )
            intex_tab += 1
        
        # Set default value to first tab if there are any tabs
        default_value = f'tab-{list(data_arrays.keys())[0]}' if data_arrays else None
        return tabs, default_value
    else:
        raise dash.exceptions.PreventUpdate

# First create a function to get the table IDs
def get_table_outputs(selected_file):
    """Get list of Output objects for all tables in the HDF5 file"""
    if selected_file and selected_file.endswith('.hdf5'):
        data_arrays = utils.load_dataframe_hdf5(os.path.join(UPLOAD_DIRECTORY, selected_file))
        return [Output(f'table-{table_name}', 'data') for table_name in data_arrays.keys()]
    return []

# Then modify the callback to use dynamic outputs
@app.callback(
    Output('hdf5-data-tables-container', 'children'),
    [Input('output-selected-file', 'children'),
     Input('dropdown-groups', 'value'),
     Input('checklist-motions', 'value'),
     Input('checklist-weights', 'value')],
    prevent_initial_call=True
)
def update_data_table_hdf5(selected_file, selected_groups, selected_motions, selected_weights):
    if not all([selected_file, selected_groups, selected_motions, selected_weights]):
        raise dash.exceptions.PreventUpdate
        
    if not selected_file.endswith('.hdf5'):
        raise dash.exceptions.PreventUpdate
        
    data_arrays = utils.load_dataframe_hdf5(os.path.join(UPLOAD_DIRECTORY, selected_file))
    updated_tables = []
    
    for table_name, df in data_arrays.items():
        filtered_df = df[df["group"].isin(selected_groups)]
        filtered_df = filtered_df[filtered_df["motion"].isin(selected_motions)]
        
        if 'weight' in filtered_df:
            filtered_df = filtered_df[filtered_df["weight"].isin(selected_weights)]
            
        filtered_df.reset_index(drop=True, inplace=True)
        updated_tables.append(filtered_df.to_dict('records'))
    
    return updated_tables
