import os

import dash
import plotly.express as px
import utils.utils as utils

from app import app
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
from dash import dash_table, dcc, html

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

@app.callback(
    [Output('area_under_radius', 'figure'),
     Output('mov_avg_radius', 'figure'),
     Output('shared_triggers', 'data'), 
     Output('shared_limits', 'data')],
    [Input('hdf5-data-tabs', 'children'),
     Input('triggers-data', 'data'),
     Input('area_under_radius', 'relayoutData'),
     Input('mov_avg_radius', 'relayoutData'),
     Input("start-trigger-value-input", "value"),
     Input("stop-trigger-value-input", "value")],
    [State("subplot-selector-triggers", "value"),
     State('shared_triggers', 'data'),
     State('shared_limits', 'data'),
     State('area_under_radius', 'figure'),
     State('mov_avg_radius', 'figure')],
    prevent_initial_call=True)

def update_figures(hdf5_experimentation_data, triggers_data,
                   area_relayout, avg_relayout,
                   start_trigger_value, stop_trigger_value, subplot_id,
                   shared_triggers, shared_limits, area_fig, avg_fig):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.') 

    # Handle initial load or data updates
    if trigger_id[0] in ['hdf5-data-tabs', 'triggers-data']:
        if not hdf5_experimentation_data:
            raise dash.exceptions.PreventUpdate
            
        df_list = hdf5_experimentation_data
        if not df_list:
            return px.scatter(), px.scatter(), shared_triggers, shared_limits

        num_group_subplots = len(df_list)
        triggers_data = triggers_data[0]
        limits_data = {
            'alarm_high': 44,
            'alarm_low': 36,
            'warning_low': 38,
            'target': 40,
            'warning_high': 42,
            'goal_t': 180,
            'freq': 100
        }
        print('-----------------BEFORE PLOTTING DATA -----------------')
        extracted_df = utils.extract_data(df_list, triggers_data, limits_data)
        file_names = ["Test run " + name[:-4] for name, _ in extracted_df.items()]

        fig_area_all, fig_mov_avg_all = utils.create_graph_config_general(num_group_subplots, file_names)
        # Generate figures with shared triggers
        fig_area_all = utils.get_fig_area(extracted_df, fig_area_all, shared_triggers, shared_limits)
        fig_mov_avg_all = utils.get_fig_avg(extracted_df, fig_mov_avg_all, shared_triggers, shared_limits)
        return fig_area_all, fig_mov_avg_all, shared_triggers, shared_limits
    
    elif trigger_id[0] in ['apply-changes-triggers-modal'] and subplot_id and start_trigger_value and stop_trigger_value:
        shared_triggers["triggers"][subplot_id]["start"] = start_trigger_value
        shared_triggers["triggers"][subplot_id]["stop"] = stop_trigger_value
        area_fig, avg_fig = utils.update_figures(hdf5_experimentation_data, shared_triggers, shared_limits)
        return area_fig, avg_fig, shared_triggers, shared_limits
    
    # Handle shape updates through relayoutData
    elif trigger_id[0] in ['area_under_radius', 'mov_avg_radius'] and 'relayoutData' in trigger_id[1]:
        relayout_data = area_relayout if trigger_id[0] == 'area_under_radius' else avg_relayout
        if relayout_data and list(relayout_data.keys()) == ['autosize']:
            return area_fig, avg_fig, shared_triggers, shared_limits

        # Handle shape movement
        if relayout_data and any('shapes' in key for key in relayout_data.keys()):
            modified_shapes = {}  # Track which shapes are modified
            
            # First collect all modifications
            for key in relayout_data:
                if 'shapes' in key:
                    shape_num = int(key.split('[')[1].split(']')[0])
                    if shape_num not in modified_shapes:
                        modified_shapes[shape_num] = {}
                    
                    if 'x0' in key or 'x1' in key:
                        new_x = relayout_data[key]
                        modified_shapes[shape_num]['x'] = new_x
                    elif 'y0' in key or 'y1' in key:
                        new_y = relayout_data[key]
                        modified_shapes[shape_num]['y'] = new_y
     
            # Apply all modifications at once
            for shape_num, changes in modified_shapes.items():
                subplot_idx = shape_num // 7
                shape_idx = shape_num - 7 * subplot_idx
                # Update both figures
                for fig in [area_fig, avg_fig]:
                    if 'layout' in fig and 'shapes' in fig['layout']:
                        # Update shared triggers for vertical lines
                        if shape_idx < 2:  # First two shapes are vertical lines
                            fig['layout']['shapes'][shape_num].update({
                                'x0': changes['x'],
                                'x1': changes['x'],
                                'visible': True,
                                'editable': True
                            })
                            trigger_type = 'start' if shape_num % 2 == 0 else 'stop'
                            shared_triggers['triggers'][str(subplot_idx)][trigger_type] = changes['x']
                            # Update shared limits for horizontal lines
                        elif shape_idx >= 2:  # Horizontal lines
                            fig['layout']['shapes'][shape_num].update({
                            'y0': changes['y'],
                            'y1': changes['y'],
                            'visible': True,
                            'editable': True
                            })
                            limit_types = ['alarm_low', 'alarm_high', 'warning_low', 'target', 'warning_high']
                            shared_limits['limits'][str(subplot_idx)][limit_types[shape_idx - 2]] = changes['y']
            area_fig, avg_fig = utils.update_figures(hdf5_experimentation_data, shared_triggers, shared_limits)
            return area_fig, avg_fig, shared_triggers, shared_limits
    return area_fig, avg_fig, shared_triggers, shared_limits

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
               Output('dropdown-groups', 'value')],
              [Input('output-selected-file', 'children'),],
             )
def initialize_options(selected_file):
    if selected_file:
        df_list = utils.load_dataframe_hdf5(os.path.join(UPLOAD_DIRECTORY, selected_file)) #load_dataframe
        return df_list,\
               df_list.keys()
    else:
        raise dash.exceptions.PreventUpdate

@app.callback(
    [Output('hdf5-data-tabs', 'children'),
     Output('hdf5-data-tabs', 'value')],
    [Input('output-selected-file', 'children')],
    prevent_initial_call=True
)
def initialize_options_hdf5(selected_file):

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

@app.callback(
    Output("subplot-selector-function", "options"),
    Output("subplot-selector-triggers", "options"),
    [Input("open-function-modal", "n_clicks"),
     Input("open-triggers-modal", "n_clicks")],
    [State("shared_triggers", "data")],
    prevent_initial_call=True
)
def subplot_options(n_clicks_func, n_clicks_trig, shared_triggers):
    if not (n_clicks_func or n_clicks_trig):
        return [], []  
    if not shared_triggers or 'triggers' not in shared_triggers:
        return [], []   
    # Convert subplot indices to dropdown options format
    options = [{'label': f'Subplot {int(x) + 1}', 'value': x} 
               for x in shared_triggers['triggers'].keys()]
    return options, options

@app.callback(
    Output("function-modal", "is_open"),
    [Input("open-function-modal", "n_clicks"),
     Input("close-function-modal", "n_clicks"),
     Input("apply-changes-function-modal", "n_clicks")],
    [State("function-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal_functions(n_open, n_close, n_apply, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "open-function-modal" and n_open:
        return True
    elif trigger_id in ["close-function-modal", "apply-changes-function-modal"]:
        return False
    return is_open

@app.callback(
    Output("triggers-modal", "is_open"),
    [Input("open-triggers-modal", "n_clicks"),
     Input("close-triggers-modal", "n_clicks"),
     Input("apply-changes-triggers-modal", "n_clicks")],
    [State("triggers-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal_triggers(n_open, n_close, n_apply, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "open-triggers-modal" and n_open:
        return True
    elif trigger_id in ["close-triggers-modal", "apply-changes-triggers-modal"]:
        return False
    return is_open

@app.callback(
   [Output("start-trigger-value-input", "value"),
    Output("stop-trigger-value-input", "value")],
   [Input("select-subplot-triggers", "n_clicks")],
   [State("subplot-selector-triggers", "value"),
    State("shared_triggers", "data")],
   prevent_initial_call=True
)
def get_trigger_new_values(subplot_selected, subplot_id, shared_triggers):
    if subplot_selected and subplot_id and shared_triggers:
        start_val = shared_triggers["triggers"][subplot_id]["start"]
        stop_val = shared_triggers["triggers"][subplot_id]["stop"]
        return start_val, stop_val
    return None, None

@app.callback(
    Output('feature-tab-content', 'children'),
    [Input('hdf5-data-tabs', 'children'),
     Input('triggers-data', 'data'),
     Input('feature-tab-content', 'relayoutData')],
    [State('shared_triggers', 'data'),
     State('shared_limits', 'data'),
     State('feature-tab-content', 'children')],
    prevent_initial_call=True)

def update_figure_features(hdf5_exp_data, triggers_data, features_relayout,
                   shared_triggers, shared_limits, features_fig):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    window_size = 50
    trigger_id = ctx.triggered[0]['prop_id'].split('.') 

    # Handle initial load or data updates
    if trigger_id[0] in ['hdf5-data-tabs', 'triggers-data']:
        if not hdf5_exp_data:
            raise dash.exceptions.PreventUpdate

        num_group_subplots = len(hdf5_exp_data)
        triggers_data = triggers_data[0]
        limits_data = {
            'alarm_high': 44,
            'alarm_low': 36,
            'warning_low': 38,
            'target': 40,
            'warning_high': 42,
            'goal_t': 180,
            'freq': 100
        }
        print('-----------------BEFORE PLOTTING DATA 2-----------------')
        extracted_df = utils.extract_data(hdf5_exp_data, triggers_data, limits_data)
        file_names = ["Test run " + name[:-4] for name, _ in extracted_df.items()]
        fig_features_dict = utils.create_graph_config_features(num_group_subplots, file_names, 2)
        fig_features_dict = utils.get_fig_features(extracted_df, fig_features_dict, window_size)
    
        feature_graphs = [
        dcc.Graph(
            id=f'features_extraction_{key}',
            figure=fig,
            config=utils.create_graph_config(),
            animate=False,
            style={'height': '800px', 'marginBottom': '40px'}
        )
        for key, fig in fig_features_dict.items()
        ]
        return html.Div(feature_graphs)
        
    # Handle shape updates through relayoutData
    elif trigger_id[0] in ['features_extraction'] and 'relayoutData' in trigger_id[1]:
        relayout_data = features_relayout
        if relayout_data and list(relayout_data.keys()) == ['autosize']:
            return features_fig
    raise dash.exceptions.PreventUpdate
