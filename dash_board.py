import dash_bootstrap_components as dbc
from dash import html, Dash, Output, Input, dcc
from dash.exceptions import PreventUpdate
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import pickle

prediction_model=pickle.load(open('default_prediction_model.pkl','rb'))
prediction_features=['loans_credit_limit', 'days_since_opened', 'days_since_confirmed', 'loans_credit_cost_rate', 'record_number', 'utilization']

input_file=pd.read_csv('DataToPredict.csv')
input_file.set_index(input_file['customer_id'],inplace=True)
input_for_prediction=input_file[prediction_features]
input_for_prediction=input_for_prediction.apply(lambda x: pd.to_numeric(x,downcast='float',errors='coerce')).fillna(-1)
input_file['primary_close_flag']=prediction_model.predict(input_for_prediction)


def inject_inplace(src, dst):
    for attr in dir(dst):
        try:
            setattr(dst, attr, getattr(src, attr))
        except AttributeError:
            pass
        except NotImplementedError:
            pass


# Create explainer to ensure fast initial load.
explainer=ClassifierExplainer(prediction_model, input_file[prediction_features],input_file['primary_close_flag'] ,idxs=input_file.index)
dashboard=ExplainerDashboard(explainer,title='Default Prediction',model_summary=False,shap_dependence=False,shap_interaction=False,contribution=False)

# Setup app with (hidden) dummy classifier layout.
dummy_layout = html.Div(dashboard.explainer_layout.layout(), style=dict(display="none"))
app = Dash(__name__,url_base_pathname='/dash/')  
app.config.external_stylesheets = [dbc.themes.BOOTSTRAP]
app.layout = html.Div([
    html.Button('Submit', id='submit', n_clicks=0),
    dcc.Loading(html.Div(id='container', children=dummy_layout), fullscreen=True)
])
# Register the callback before the app starts.
dashboard.explainer_layout.register_callbacks(app)


@app.callback(Output('container', 'children'), Input('submit', 'n_clicks'))
def load_complete_dataset(n_clicks):
    if n_clicks != 1:
        raise PreventUpdate

    # Replace in-memory references to the full dataset to sure callbacks target the full dataset.
    full_explainer = ClassifierExplainer(prediction_model, input_file[prediction_features],input_file['primary_close_flag'] ,idxs=input_file.index)
    inject_inplace(full_explainer, explainer)
    return ExplainerDashboard(explainer,model_summary=False,shap_dependence=False,shap_interaction=False,contribution=False).explainer_layout.layout()


if __name__ == "__main__":
    app.run_server(debug=False)