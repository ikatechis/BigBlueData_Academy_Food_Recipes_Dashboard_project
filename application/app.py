# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:58:29 2021

@author: iason
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc
from dash import html
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
from dash.dependencies import Input, Output, State



# ------------------------------------------------------ GRAPHS ------------------------------------------------------
# Define Global Variables
df_ing = pd.read_csv('../data/ingredient_data.csv')
df_rec = pd.read_csv('../data/recipes_per_year.csv')

cuisine_opt = ['North American', 'Italian', 'European', 'Asian', 'South West Pacific', 'French', 'Indian', 
               'Greek', 'Chinese', 'Mexican', 'Thai', 'German', 'Spanish', 'Middle Eastern', 'South American', 
               'Japanese', 'Moroccan', 'Irish', 'African', 'Scandinavian', 'Scottish', 'Ontario', 'Russian', 
               'Pacific Northwest', 'Vietnamese', 'Australian', 'Polish', 'Swedish', 'Korean', 'Swiss', 'Turkish', 
               'Portuguese', 'Hungarian', 'South African', 'Persian', 'Filipino', 'Indonesian', 'English', 
               'Egyptian', 'Szechuan', 'Puerto Rican', 'Ethiopian', 'Caribbean']

time_of_day_opt = ['Lunch', 'Dinner', 'Snack', 'Breakfast', 'Brunch']

menu_opt = ['Main Dish', 'Dessert', 'Side Dish', 'Appetizer', 'Salad', 'Finger Food']

type_opt = ['Snack', 'Pasta', 'Stew', 'Cake', 'Cookies', 'Crock Pot', 'Pie', 'Bread', 'Beverage', 'Dip', 
            'No Cook', 'Frozen', 'Cupcakes', 'Grill', 'Roast', 'Pizza', 'Steak', 'Sauce', 'Stir Fry', 'Deep Fry', 
            'Broil/Grill', 'Soup', 'Sandwich', 'Burger']

special_opt = ['Party', 'Weeknight', 'Gifts', 'Picnic', 'Romantic', 'Thanksgiving', 'Barbecue', 'Easter', 
               'Christmas', 'Valentines Day', 'Wedding', 'Halloween', 'Camping', 'Birthday', 'New Years', 
               'St Patricks Day', 'Mothers Day', 'April Fools Day']

health_opt = ['Kid Friendly', 'Toddler Friendly', 'Baby Friendly', 'Sugar Free', 'Low Sugar', 'Low Sodium', 
              'Low Protein', 'High Protein', 'Vegetarian', 'Vegan', 'No Meat', 'Low Fat', 'Low Saturated Fat', 
              'Fat Free', 'Low Carb', 'Very Low Carb', 'Low Calorie', 'Lactose Free', 'High Calcium', 
              'Healthy', 'Gluten Free', 'Egg Free', 'Diabetic Friendly', 'Low Cholesterol']

other_opt = ['Light', 'Easy', 'Spicy', 'Served Hot', 'Served Cold', 'Summer', 'Fall', 'Winter', 'Spring', 
             '< 60 Mins', '< 30 Mins', '< 4 Hours', '< 15 Mins', '> 1 Day', 'Kid Friendly', 'Toddler Friendly', 
             'Baby Friendly', 'College', 'Comfort Food']

rec = pd.read_csv('../data/cleaned_recipes_fotis.csv')


a = pd.read_csv('../data/total_recipes_per_year_fotis.csv')

# Functions
def make_sunburst(min_rec=100, max_rec=df_ing.counts.max()):
    
    colorz = ['#8dd3c7','#b3de69','#bebada','#fb8072','#80b1d3','#fdb462',
              '#ffffb3','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']

    
    sun = df_ing[(df_ing.counts >= min_rec) & (df_ing.counts <= max_rec)]
    fig = px.sunburst(sun, path=['food_group', 'food_subgroup', 'ingredients'],
                      values='counts', height=600, 
                      color_discrete_sequence=colorz,
                     )
    fig.update_layout(
                        {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)'}, 
                      margin={"r":0,"t":0,"l":0,"b":0}
                      )
    return fig

sun_slider = dcc.RangeSlider(
            id='sun-slider',
            min=1,
            max=df_ing['counts'].max(),
            
            value=[100, df_ing['counts'].max()],
            step=10,
            
            tooltip={"placement": "bottom", "always_visible": False},
            marks={
        
            100: {'label': '100'},
            10000: {'label': '10k'},
            50000: {'label': '50k'},
            100000: {'label': '100k', 'style': {'color': '#f50'}},
            int(f'{df_ing["counts"].max()}'): {'label': 'max'}
        }
        )
# colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3',
#           '#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd',
#           '#ccebc5','#ffed6f']

def make_funnel(min_rec=100, max_rec=df_ing.counts.max(), from_sun='ALL'):
    if from_sun == 'ALL':
        data = df_ing[(df_ing.counts >= min_rec) & (df_ing.counts <= max_rec)]
    elif type(from_sun) == list:
        data = df_ing[df_ing[from_sun[0]] == from_sun[1]]

    data = data.iloc[:10] # Choose the 5 top ingredients
    perc = data.percentages.apply(lambda x: str(x) + ' %')
    perc.name = 'percent'
    data = pd.concat([data, perc], axis=1)
    fig = go.Figure(go.Funnel(
                y = data.ingredients,
                x = data.counts,
                textposition = "inside",
                text = data.percent,
                opacity=0.8, 
                marker={"color": ['#fb8072', '#bc80bd', '#fdb462', 
                                  '#80b1d3', '#8dd3c7', '#bebada', 
                                  '#fccde5', '#d9d9d9', '#ccebc5', 
                                  '#ffffb3'],
                },
                connector={"line": {"color": "royalblue", 
                                    "dash": "dot", "width": 1}})
    )

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
                      margin={"r":5,"t":5,"l":5,"b":5}, xaxis_title='Year',font={'size':16})

    return fig


def make_ingr(from_sun='ALL'):
    if from_sun == 'ALL':
        data = df_ing
    elif type(from_sun) == list:
        data = df_ing[df_ing[from_sun[0]] == from_sun[1]]
    data = data.iloc[:5] # Choose the 5 top ingredients
    ing = data.T
    ing.columns = ing.iloc[0]
    ing1 = ing.iloc[4:24].reset_index().rename(columns={'index': 'year'})
    melt = ing1.melt(id_vars='year', value_name='recipes_number')
    ing_per = ing1.iloc[:, 1:].apply(lambda x: x/df_rec.nr_recipes.iloc[:-1]*100).fillna(0)
    ing_per = ing_per.round(1)
    ing_per['year'] = ing1.year
    melt_perc = ing_per.melt(id_vars='year', value_name='recipes_percentage')
    melt['recipes_percentage'] = melt_perc['recipes_percentage']
    melt['recipes_number'] = melt.loc[:, 'recipes_number'].astype(int)
    
    colors = ['#8dd3c7','#bc80bd','#fb8072','#b3de69','#fdb462']

    fig = px.scatter(melt, x="year", y="recipes_percentage", size='recipes_number', 
                     color='ingredients', size_max=40, template='ggplot2', color_discrete_sequence=colors)
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
        margin={"r":0,"t":0,"l":0,"b":0}, yaxis_title='Recipes %', xaxis_title='Year')
    
    return fig

def make_time_figs(dropped='rec'):
    print('HEEEEEERE')
    
    if dropped == 'rec':
        fig = px.area(df_rec, x='year', y='nr_recipes', color_discrete_sequence=['#bc80bd'], template='ggplot2')
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 
       'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
            margin={"r":0,"t":0,"l":0,"b":0}, yaxis_title='Number of Recipes', xaxis_title='Year')
        return fig
    
    elif dropped == 'ing':
        df1 = df_rec
        fig = go.Figure([
                go.Scatter(
                    name='Average',
                    x=df1['year'],
                    y=df1['ingredients_avrg'],
                    mode='lines',
                    line=dict(color='#fb8072', width=4),
                ),
                go.Scatter(
                    name='Upper Bound',
                    x=df1['year'],
                    y=df1['ingredients_avrg']+df1['ingredients_std'],
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                ),
                go.Scatter(
                    name='Lower Bound',
                    x=df1['year'],
                    y=df1['ingredients_avrg']-df1['ingredients_std'],
                    marker=dict(color='#fb8072'),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(252, 205, 229, 0.4)',
                    fill='tonexty',
                    showlegend=False
                )
            ])
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 
       'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
            xaxis_title='Year',
            yaxis_title='Number of Ingredients',
            title='Average number of Ingredients per Recipe',
            hovermode="x", template='ggplot2',
            )
        return fig
    
    elif dropped == 'cal':
        df1 = df_rec
        fig = go.Figure([
                go.Scatter(
                    name='Average',
                    x=df1['year'],
                    y=df1['calories_median'],
                    mode='lines',
                    line=dict(color='#fb8072', width=4),
                ),
                go.Scatter(
                    name='Upper Bound',
                    x=df1['year'],
                    y=df1['calories_median']+df1['calories_std'],
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                ),
                go.Scatter(
                    name='Lower Bound',
                    x=df1['year'],
                    y=df1['calories_median']-df1['calories_std'],
                    marker=dict(color='#fdb462'),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(255, 255, 179, 0.5)',
                    fill='tonexty',
                    showlegend=False
                )
            ])
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 
       'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
        xaxis_title='Year',
        yaxis_title='Calories',
        title='Median Calories per Recipe',
        hovermode="x", template='ggplot2',
    )
    return fig

# ------------------------------------------------------ APP ------------------------------------------------------
markdown_text = '''
### An Interactive Dashboard
'''



tab1 = html.Div([
     
   html.Div([ # row 1
       html.Div([dcc.Graph(id='sun-fig'), 
             sun_slider], id='sunburst'),
       html.Div([html.H3(id='fun-title', children="Most Common Ingregients"),
                 dcc.Graph(id='funnel-fig'), 
             ], id='funnel')
       
       ]
       ,  className='tab1_rows'),
       
   
   html.Div([
       html.Div([dcc.Graph(id='ingr-fig'), 
         ], id='ingr'),
             
       
       html.Div([dcc.Dropdown(id='time-drop',
                    options=[
            {'label': 'Number of Recipes', 'value': 'rec'},
            {'label': 'Average Number of Ingredients', 'value': 'ing'},
            {'label': 'Average Calories', 'value': 'cal'}
            ],
            value='rec',
            clearable=False, searchable=False,
            ),
                 dcc.Graph(id='time-fig'), 
             ], id='time')
       ], id='row2', className='tab1_rows'),


], 
className='tab1')


tab2 = html.Div([
    html.H3('Tab content 2')
], className='tab2')

app = dash.Dash(__name__)
app.title = 'Recipes Dashboard'
app._favicon = ("burger.ico")

app.layout = html.Div(id='main', children=[
    
        html.Div(
        [
            html.H1(id='title', children="The Recipes of food.com"),
            dcc.Markdown(children=markdown_text)
               
            ],
            className="header",
        ),
        
        html.Div([
            dcc.Tabs(
       id="tabs-with-classes",
       value='tab-1',
       parent_className='custom-tabs',
       className='custom-tabs-container',
       children=[
           dcc.Tab(
               label='Explore the Ingredients',
               value='tab-1',
               className='custom-tab',
               selected_className='custom-tab--selected'
           ),
           dcc.Tab(
               label='Explore the Recipes',
               value='tab-2',
               className='custom-tab',
               selected_className='custom-tab--selected'
           ),
       ]),
   html.Div(id='tabs-content-classes')
])
         
])


# --------------------------------------------------CALLBACKS------------------------------------------------------

@app.callback(Output('tabs-content-classes', 'children'),
              Input('tabs-with-classes', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return tab1
    elif tab == 'tab-2':
        return tab2
    
@app.callback(
    Output('sun-fig', 'figure'),
    Input('sun-slider', 'value'))

def update_figure(value):
    sun = make_sunburst(value[0], value[1])
    # fun = make_funnel(value[0], value[1])
    return sun


@app.callback(
   Output("funnel-fig", "figure"), 
   Output('ingr-fig', 'figure'),
   Input("sun-fig", "clickData"),
)
def callback_func(clickData):
    click_path = "ALL"
    path=['food_group', 'food_subgroup', 'ingredients']

    if clickData:
        click_path = clickData["points"][0]["id"].split("/")
        selected = dict(zip(path, click_path))
        print(selected)
        
        percent = (clickData["points"][0]).get("percentEntry")
        print('PERCENT: '+str(percent))
    
        if 'food_subgroup' in selected and percent != 1:
            print(['food_subgroup', selected['food_subgroup']])
            click_path = ['food_subgroup', selected['food_subgroup']]

            
            
        elif 'food_subgroup' in selected and percent == 1:
            print(['food_group', selected['food_group']])
            click_path = ['food_group', selected['food_group']]

            
        elif 'food_subgroup' not in selected and percent != 1:
            print(['food_group', selected['food_group']])
            click_path = ['food_group', selected['food_group']]

        else:
            click_path = 'ALL'
 
    
    return make_funnel(from_sun=click_path), make_ingr(click_path)
        
@app.callback(
    Output('time-fig', 'figure'),
    Input('time-drop', 'value'))

def update_time(value):
    print(value)
    
    return make_time_figs(value)
  


if __name__ == '__main__':
    app.run_server(port=8050, host='127.0.0.1')