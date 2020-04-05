

from textwrap import dedent
import base64
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State

from sklearn import datasets 
import numpy as np
import pandas as pd
from Clustering_Ensemble import CluterEnsemble


from utils.figures import serve_clustering_plot, serve_metrics
import utils.dash_reusable_components as drc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

def generate_data(dataset):        
    print("generate data")
    n_samples = 200
    random_state = 170
    if dataset == 'circles':
        return datasets.make_circles(
            n_samples=n_samples,
            factor=0.5,
            noise=0.05,
            random_state=random_state
        )
    elif dataset == 'moons':
        return datasets.make_moons(
            n_samples=n_samples,
            noise=0.05,
            random_state=random_state
        )
    elif dataset == 'anistropicly':
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
        return (X, y)
    elif dataset == 'variance':
        return datasets.make_blobs(
            n_samples=n_samples,
            cluster_std=[1.0, 2.5, 0.5],
            random_state=random_state
        )
    else:
        X = np.asarray(pd.read_csv(f"Datasets/{dataset}.txt", header=None, index_col=None, sep=','))
        y = np.asarray(pd.read_csv(f"Datasets/{dataset}_label.txt", header=None, index_col=None, sep=','))
        y = np.ravel(y)
        return X, y

def read_data_from_file(filename, contents):
    print("read from file")
    if contents != None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = None
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        X = np.asarray(df.iloc[:, 0:df.shape[1]-1])
        y = np.asarray(df.iloc[:, -1])
        return X, y


app.layout = html.Div(style={'backgroundColor': '#282b38', 'margen':'0px' }, children=[

    html.Div(className="banner", children=[
        html.Div(className='container scalable', children=[
            html.H2(html.A(
                'Clustering Ensemble Explorer',
                href='https://github.com/Yanis2016/Clustering-Ensemble',
                style={
                    'text-decoration': 'none',
                    'color': 'inherit'
                }
            )),

            html.A(
                html.Img(src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"),
                href='https://plot.ly/products/dash/'
            )
        ]),
    ]),
    html.Div(id='body', className='container scalable2', children=[

        html.Div(className='row', children=[
            html.Div(
                className='three columns',
                style={
                    'min-width': '24.5%',
                    'max-height': 'calc(100vh - 85px)',
                    'overflow-y': 'auto',
                    'overflow-x': 'hidden',
                    'font': '#7FDBFF',
                },
                children=[
                    drc.Card([
                        drc.NamedDropdown(
                            name='Select Dataset',
                            id='dropdown-select-dataset',
                            options=[
                                {'label': 'Moons', 'value': 'moons'},
                                {'label': 'Circles', 'value': 'circles'},
                                {'label': 'Anisotropicly distributed data', 'value' : 'anistropicly'},
                                {'label': 'Data with varied variances', 'value': 'variance'},
                                {'label': 'Pathbased', 'value': 'pathbased'},
                                {'label': 'Aggregation', 'value': 'aggregation'},
                                {'label': 'Flame', 'value': 'flame'}
                            ],
                            clearable=False,
                            searchable=False,
                            value='moons'
                        ),

                        html.Div(
                            id='div-upload-data',
                            children=[
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div(
                                        html.A('Select a CSV or excel file')),
                                    style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px 0px',
                                        'border-color': '#8f9197'
                                    },
                                    multiple=False,
                                    contents = None
                                ),
                            ]
                        ),
    
                        html.Div(
                            id='div-target',
                            children=[
                                drc.NamedRadioItems(
                                    name='Target included in the file',
                                    id='target',
                                    options=[
                                        {'label': 'Yes', 'value': 'Yes'},
                                        {'label': 'No', 'value': 'No'}
                                    ],
                                    value='Yes',
                                    style={'display': 'none'}
                                )
                            ]       
                        ),

                    ]),

                    drc.Card([
                        drc.NamedSlider(
                            name='Number of clusters',
                            id='n_clusters',
                            min=2,
                            max=20,
                            step=1,
                            marks={i: str(i) for i in range(2, 21)},
                            value=2
                        ),

                        drc.NamedSlider(
                            name='Number of base partitions',
                            id='n_partitions',
                            min=100,
                            max=1000,
                            step=100,
                            marks={i: str(i) for i in range(100, 1001, 100)},
                            value=500
                        ),

                        drc.NamedRadioItems(
                            name='Number of clusters for each partitions',
                            id='k_type',
                            options=[
                                {'label': 'Fixed', 'value': 'Fixed'},
                                {'label': 'Random', 'value': 'Random'}
                            ],
                            value='Fixed'
                        ),
                        drc.NamedRadioItems(
                            name='Consensus validation methods',
                            id='cons_validation',
                            options=[
                                {'label': 'Average confidence', 'value': 'ac'},
                                {'label': 'Average neighborhood confidence', 'value': 'anc'},
                                {'label': 'Average dynamic neighborhood confidence', 'value': 'adnc'}
                            ],
                            value='ac'
                        ),
                    ]),

                    drc.Card([
                        html.Button(
                            'run',
                            id='run',
                            n_clicks=0,
                            style={
                                'margin': '10px 10px', 'color' :'#8f9197',
                            }
                        ),
                    ]),


                    html.Div(
                        dcc.Markdown(dedent("""
                        [Click here](https://github.com/plotly/dash-svm) click here to visit the project, and find out more about Clustering Ensemble.
                        """)),
                        style={'margin': '20px 0px', 'text-align': 'center', 'color' :'#8f9197'}
                    )
                ]),

                html.Div(
                    id='div-graphs',
                    children=dcc.Graph(
                        id='graph-ensemble-clustering',
                        style={'display': 'none'}
                    )
                ),
            ]),
        ]),
])

@app.callback(Output('n_clusters', 'value'),
              [Input('dropdown-select-dataset', 'value')])
def update_n_clusters(dataset):
    
    if dataset in {'moons', 'circles', 'flame'} :
        return 2
    if dataset in {'anistropicly', 'variance', 'pathbased'}:
        return 3
    if dataset == 'aggregation':
        return 7

@app.callback(Output('div-upload-data', 'children'),
              [Input('dropdown-select-dataset', 'value')])
def update_target_off(value):
    print("update target off")
    return  dcc.Upload(
                id='upload-data',
                children=html.Div(
                    html.A('Select a csv file')),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px 0px',
                    'border-color': '#8f9197'
                },
                multiple=False,
                contents = None
            ),

  

@app.callback(Output('div-target', 'children'),
              [Input('upload-data', 'contents')])
def update_target_on(contents):
    print("update target on")
    if contents == None:
        return drc.NamedRadioItems(
                    name='Target included in the file',
                    id='target',
                    options=[
                        {'label': 'Yes', 'value': 'Yes'},
                        {'label': 'No', 'value': 'No'}
                    ],
                    value='Yes',
                    style={'display': 'none'}
                )
            
    return  drc.NamedRadioItems(
                name='Target included in the file',
                id='target',
                options=[
                    {'label': 'Yes', 'value': 'Yes'},
                    {'label': 'No', 'value': 'No'}
                ],
                value='Yes',
            )


@app.callback(Output('div-graphs', 'children'),
              [Input('run', 'n_clicks')],
              [State('dropdown-select-dataset', 'value'),
               State('upload-data', 'filename'),
               State('upload-data', 'contents'),
               State('n_clusters', 'value'),
               State('n_partitions', 'value'),
               State('k_type', 'value'),
               State('cons_validation', 'value'),
               State('target', 'value')])
def update_clustering_graph(n_clicks,
                            dataset,
                            filename,
                            contents,
                            n_clusters,
                            n_partitions,
                            k_type,
                            cons_validation,
                            target):
    print("update clustering graph")
    if n_clicks == 0:
        return
    X,y_pred = None, None

    if contents == None:
        X, y = generate_data(dataset)
    else :
        try:
            X, y = read_data_from_file(filename, contents)
        except :
            return dcc.Markdown(
                '''
                **There was an error processing this file. Please note that only CSV and Excel
                  files are accepted, with or without target and without header and without index. Retry**',
                ''',
                style={'padding': '100px 100px 20px 200px', 'color': 'red'}
            )

    model = CluterEnsemble(n_clusters=n_clusters,
                           n_partitions=n_partitions,
                           k_type=k_type,
                           cons_validation=cons_validation
                           )
    y_pred = model.fit_predict(X)
    clustering_figure = serve_clustering_plot(X, y_pred)
    evaluation_figure = serve_metrics(X, y_pred, y)

    df = pd.DataFrame(np.concatenate([X, y_pred.reshape((y_pred.shape[0], 1))], axis=1))
    df.iloc[:,-1] = pd.Series(df.iloc[:, -1].values, dtype=int)
    csv_string = df.to_csv(index=False, header=False, encoding='utf-8')
    href = "data:text/csv;charset=utf-8,%EF%BB%BF" + csv_string
   
    if target == "No":
        return [
           html.Div(
                className='six columns',
                style={
                    'margin-top': '5px',
                    'margin-left': '50px',
                },
                children=[
                    html.A(
                        'Download data with target',
                        id='download-link',
                        download="rawdata.csv",
                        href=href,
                        target="_blank"
                    ),
                    
                    dcc.Graph(
                        id='graph-ensemble-clustering',
                        figure=clustering_figure,
                        style={
                            'height': 'calc(86.5vh - 90px)',

                        }
                    )
                ]
            ),
        ]
    else:
        return [
            html.Div(
                className='six columns',
                style={
                    'margin-top': '5px',
                    'margin-left': '50px',
                },
                children=[
                    html.A(
                        'Download data with target',
                        id='download-link',
                        download="rawdata.csv",
                        href=href,
                        target="_blank"
                    ),
                    
                    dcc.Graph(
                        id='graph-ensemble-clustering',
                        figure=clustering_figure,
                        style={
                            'height': 'calc(86.5vh - 90px)',

                        }
                    )
                ]
            ),

            html.Div(
                className='six coms',
                style={
                    'margin-top': '10px',
                    'margin-left': '80%',
                    'margin-right': '10px',
                    'width': '20.5%',


                    # Remove possibility to select the text for better UX
                    'user-select': 'none',
                    '-moz-user-select': 'none',
                    '-webkit-user-select': 'none',
                    '-ms-user-select': 'none'
                },
                children=[
                    dcc.Graph(
                        id='graph_metrics',
                        style={'fontColor': 'blue','height': '50%'},

                        figure=evaluation_figure
                    ),
                ]
            ),
        ]

if __name__ == '__main__':
    app.run_server(debug=True)
