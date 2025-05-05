import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import base64
import io

app = dash.Dash(__name__)
app.title = "Tip Prediction Dashboard"

# Initialize empty dataframe
df = pd.DataFrame()
model = None
features = []
categorical_features = []
numerical_features = []
target_variable = None
preprocessor = None

app.layout = html.Div([
    html.H1("Tip Prediction Dashboard", style={'textAlign': 'center'}),

    # Upload Component
    html.Div([
        html.H3("Upload File"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select a CSV File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px 0'
            },
            multiple=False
        ),
        html.Div(id='output-data-upload'),
    ], style={'margin': '20px'}),

    # Select Target Component
    html.Div([
        html.H3("Select Target:"),
        dcc.Dropdown(
            id='target-dropdown',
            options=[],
            placeholder="Select target variable...",
            style={'width': '300px', 'margin': '10px 0'}
        ),
    ], style={'margin': '20px'}),

    # Barcharts Components
    html.Div([
        html.Div([
            # Left Chart
            html.Div([
                #html.H4("Average tip by sex", style={'textAlign': 'center'}),
                dcc.RadioItems(
                    id='category-radio',
                    options=[],
                    value=None,
                    inline=True,
                    style={'margin': '10px 0', 'textAlign': 'center'}
                ),
                dcc.Graph(id='category-bar-chart')
            ], style={'width': '48%', 'display': 'inline-block'}),

            # Right Chart
            html.Div([
                html.H4(style={'textAlign': 'center'}),
                dcc.Graph(id='correlation-bar-chart')
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        ], style={'margin': '20px 0'})
    ]),

    # Feature Selection and Train Component
    html.Div([
        html.H3("Feature Selection"),
        html.Div(id='feature-checklist-container', children=[
            dcc.Checklist(
                id='feature-checklist',
                options=[],
                value=[],
                labelStyle={'display': 'block', 'margin': '5px 0'}
            )
        ]),
        html.Button("Train", id='train-button',
                  style={'margin': '10px 0', 'padding': '5px 15px'}),
        html.Div(id='train-output')
    ], style={'margin': '20px'}),

    # Predict Component
    html.Div([
        html.H3("Predict"),
        dcc.Input(
            id='prediction-input',
            type='text',
            style={'width': '300px', 'margin': '10px 0', 'padding': '5px'}
        ),
        html.Button("Predict", id='predict-button',
                  style={'margin': '10px 0', 'padding': '5px 15px'}),
        html.Div(id='prediction-output', style={'margin': '10px 0'})
    ], style={'margin': '20px'})
])

# Callback for file upload
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('target-dropdown', 'options')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    global df, categorical_features, numerical_features

    if contents is None:
        return [html.P("No file uploaded yet."), []]

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=r'[,;|]')
        else:
            return [html.Div(['Please upload a CSV file.']), []]
    except Exception as e:
        return [html.Div(['There was an error processing this file.']), []]

    # Preprocessing
    df = df.dropna(how='all')

    # Identify numerical and categorical columns
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Create dropdown options
    options = [{'label': col, 'value': col} for col in numerical_features]

    return [
        html.Div([
            html.P(filename),
            html.P(f"{df.shape[0]} rows, {df.shape[1]} columns"),
        ]),
        options
    ]

# Callback for target selection and radio items
@app.callback(
    [Output('category-radio', 'options'),
     Output('category-radio', 'value'),
     Output('feature-checklist', 'options')],
    [Input('target-dropdown', 'value')]
)
def update_target_selection(target):
    global target_variable, numerical_features, categorical_features

    if target is None or df.empty:
        return [[], None, []]

    target_variable = target

    # Update numerical features (exclude target)
    numerical_features_updated = [f for f in numerical_features if f != target]

    # Create options for category radio and feature checklist
    category_options = [{'label': col, 'value': col} for col in categorical_features]
    default_category = categorical_features[0] if categorical_features else None

    feature_options = [{'label': col, 'value': col} for col in numerical_features_updated + categorical_features]

    return [category_options, default_category, feature_options]

# Graph title
@app.callback(
    Output('category-bar-chart', 'figure'),
    [Input('category-radio', 'value'),
     Input('target-dropdown', 'value')]
)
def update_category_chart(category, target):
    if category is None or target is None or df.empty:
        return px.bar(title="Select a category to display")

    avg_df = df.groupby(category)[target].mean().reset_index()

    fig = px.bar(
        avg_df,
        x=category,
        y=target,
        color=category,
        title=f"Average '{target}' by '{category}'"
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="black"),
    )
    fig.update_traces(marker_color='cornflowerblue')
    return fig

# Correlation graph title
@app.callback(
    Output('correlation-bar-chart', 'figure'),
    [Input('target-dropdown', 'value')]
)
def update_correlation_chart(target):
    if target is None or df.empty:
        return px.bar(title="Select a target to display correlations")

    # Calculate correlations
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numerical_df.corr()[target].drop(target).abs().sort_values(ascending=True)

    fig = px.bar(
        corr,
        x=corr.values,
        y=corr.index,
        orientation='h',
        title=f"Correlation Strength of Numerical Variables with '{target}'"  # << Dynamic title
    )
    fig.update_layout(
        yaxis_title="Feature",
        xaxis_title="Correlation (Absolute Value)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="black"),
    )
    fig.update_traces(marker_color='cornflowerblue')
    return fig


# Callback for model training
@app.callback(
    Output('train-output', 'children'),
    [Input('train-button', 'n_clicks')],
    [State('feature-checklist', 'value'),
     State('target-dropdown', 'value')]
)
def train_model(n_clicks, selected_features, target):
    global model, preprocessor, features

    if n_clicks is None or not selected_features or not target:
        return ""

    try:
        # Prepare data
        X = df[selected_features]
        y = df[target]
        features = selected_features

        # Identify categorical and numerical features
        cat_features = [f for f in selected_features if f in categorical_features]
        num_features = [f for f in selected_features if f in numerical_features]

        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ])

        # Create model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        return html.Div([
            html.P(f"The R² score is: {r2:.2f}")
        ])
    except Exception as e:
        return html.Div([
            html.P("Error training model:", style={'color': 'red'}),
            html.P(str(e))
        ])

# Callback for predictions
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('prediction-input', 'value'),
     State('feature-checklist', 'value')]
)
def make_prediction(n_clicks, input_values, selected_features):
    global model

    if n_clicks is None or model is None:
        return ""

    if not input_values:
        return html.P("Please enter values to predict.", style={'color': 'red'})

    try:
        # Parse input values
        values = [x.strip() for x in input_values.split(',')]

        if len(values) != len(selected_features):
            return html.P(
                f"Expected {len(selected_features)} values, got {len(values)}. "
                f"Features: {', '.join(selected_features)}",
                style={'color': 'red'}
            )

        # Create input DataFrame
        input_data = {}
        for i, feature in enumerate(selected_features):
            # Try to convert to number if feature is numerical
            if feature in numerical_features:
                try:
                    input_data[feature] = [float(values[i])]
                except:
                    input_data[feature] = [values[i]]  # will be handled by imputer
            else:
                input_data[feature] = [values[i]]

        input_df = pd.DataFrame(input_data)

        # Make prediction
        prediction = model.predict(input_df)[0]

        return html.Div([
            html.H4(f"Predicted val is : {prediction:.2f}")
        ])
    except Exception as e:
        return html.Div([
            html.P("Error making prediction:", style={'color': 'red'}),
            html.P(str(e))
        ])

if __name__ == '__main__':
    app.run(debug=True)