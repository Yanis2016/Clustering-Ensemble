from textwrap import dedent

import dash_core_components as dcc
import dash_html_components as html


# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


# Custom Display Components
def Card(children, **kwargs):
    return html.Section(
        children,
        style=_merge({
            'padding': 20,
            'margin': 5,
            'borderRadius': 5,
            'border': 'thin lightgrey solid',

            # Remove possibility to select the text for better UX
            'user-select': 'none',
            '-moz-user-select': 'none',
            '-webkit-user-select': 'none',
            '-ms-user-select': 'none'
        }, kwargs.get('style', {})),
        **_omit(['style'], kwargs)
    )


def FormattedSlider(**kwargs):
    return html.Div(
        style=kwargs.get('style', {'color' :'#8f9197'}),
        children=dcc.Slider(**_omit(['style'], kwargs))
    )


def NamedSlider(name, **kwargs):
    return html.Div(
        style={'padding': '10px 10px 25px 4px', 'color' :'#8f9197'},
        children=[
            html.P(f'{name}:'),
            html.Div(
                style={'margin-left': '6px', 'color' :'#8f9197'},
                children=dcc.Slider(**kwargs)
            )
        ]
    )


def NamedDropdown(name, **kwargs):
    return html.Div(
        style={'margin': '10px 0px', 'color' :'#8f9197'},
        children=[
            html.P(
                children=f'{name}:',
                style={'margin-left': '3px', 'color' :'#8f9197'}
            ),

            dcc.Dropdown(**kwargs)
        ]
    )


def NamedRadioItems(name, **kwargs):
    if 'style' in kwargs:
        if 'display' in kwargs['style']:
            if kwargs['style']['display'] == 'none':
                return html.Div(
                    style={'padding': '10px 10px 20px 4px', 'color' :'#8f9197'},
                    children=[
                            html.P(children=f'{name}:', style={'display': 'none'}),
                            dcc.RadioItems(**kwargs)
                        ]         
                )
                
    return html.Div(
        style={'padding': '10px 10px 20px 4px', 'color' :'#8f9197'},
        children=[
            html.P(children=f'{name}:'),
            dcc.RadioItems(**kwargs)
        ]
    )

# Non-generic
def DemoDescription(filename, strip=False):
    with open(filename, 'r') as file:
        text = file.read()

    if strip:
        text = text.split('<Start Description>')[-1]
        text = text.split('<End Description>')[0]

    return html.Div(
            className='row',
            style={
                'padding': '15px 30px 27px',
                'margin': '45px auto 45px',
                'width': '80%',
                'max-width': '1024px',
                'borderRadius': 5,
                'border': 'thin lightgrey solid',
                'font-family': 'Roboto, sans-serif'
            },
            children=dcc.Markdown(dedent(text))
    )
def NameButton(name, **kwargs):
    return html.Div(
        children=
            html.Button(**kwargs)
        ),