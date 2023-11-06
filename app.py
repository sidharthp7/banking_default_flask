import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pickle
import explainerdashboard as expdb
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from flask import Flask,render_template,request,redirect,url_for,Response
import dash
import dash_bootstrap_components as dbc
from dash import html, Dash, Output, Input, dcc
from dash.exceptions import PreventUpdate
import io
import dash_route

prediction_model=pickle.load(open('default_prediction_model.pkl','rb'))

server=Flask(__name__)
app=Dash()
# app=dash.Dash(__name__,server=server,url_base_pathname='/dash/')
# app.config['suppress_callback_exceptions']=True

prediction_features=['loans_credit_limit', 'days_since_opened', 'days_since_confirmed', 'loans_credit_cost_rate', 'record_number', 'utilization']

@app.server.route('/file_upload',methods=['GET','POST'])
def index():

    if request.method=='POST':
        uploaded_file=request.files["uploaded_file"]
        filename_file=uploaded_file.filename
        if request.form.get("Predict_View")=='Predict & View' or request.form.get("Predict_Download")=='Predict & Download':
            if '.csv' in filename_file:
                input_file=pd.read_csv(uploaded_file)
                input_file.set_index(input_file['customer_id'],inplace=True)
                if input_file.shape[0]!=0:
                    if len(set(input_file.columns).intersection(prediction_features))==6:
                        input_for_prediction=input_file[prediction_features]
                        input_for_prediction=input_for_prediction.apply(lambda x: pd.to_numeric(x,downcast='float',errors='coerce')).fillna(-1)
                        input_file['primary_close_flag']=prediction_model.predict(input_for_prediction)
                        input_file.fillna('',inplace=True)
                        if request.form.get("Predict_View")=='Predict & View':
                            return redirect(url_for('dashboard'))
                        else:
                            return Response(input_file.to_csv(index=False),mimetype="text/csv",headers={"Content-disposition":"attachment; filename=Default_Prediction.csv"})
                    else:
                        return redirect(url_for('user_file_upload_error'))
                else:
                    return redirect(url_for('user_file_upload_error'))
            else:
                return redirect(url_for('user_file_upload_error'))
    else:
        return render_template('file_upload.html')

@app.server.route('/file_upload_error',methods=['GET','POST'])
def user_file_upload_error():
    if request.method=='GET':
        return render_template('file_upload_error.html')
    else:
        if request.form.get("Upload_File")=='Go back to File Upload':
            return redirect(url_for('index'))            
        else:
            return None

@app.server.route('/dash')
def dashboard():
        
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
    app = Dash()  
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


if __name__=="__main__":
    app.run(debug=True)