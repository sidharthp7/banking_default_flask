import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pickle
import explainerdashboard as expdb
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from flask import Flask,render_template,request,redirect,url_for,Response
import io

prediction_model=pickle.load(open('default_prediction_model.pkl','rb'))

app=Flask(__name__)

prediction_features=['loans_credit_limit', 'days_since_opened', 'days_since_confirmed', 'loans_credit_cost_rate', 'record_number', 'utilization']

@app.route('/file_upload',methods=['GET','POST'])
def index():
    if request.method=='POST':
        uploaded_file=request.files["uploaded_file"]
        filename_file=uploaded_file.filename
        if request.form.get("Predict_View")=='Predict & View' or request.form.get("Predict_Download")=='Predict & Download':
            if '.csv' in filename_file:
                input_file=pd.read_csv(uploaded_file)
                input_file['customer_id']=input_file.index
                if input_file.shape[0]!=0:
                    if len(set(input_file.columns).intersection(prediction_features))==6:
                        input_for_prediction=input_file[prediction_features]
                        input_for_prediction=input_for_prediction.apply(lambda x: pd.to_numeric(x,downcast='float',errors='coerce')).fillna(-1)
                        input_file['primary_close_flag']=prediction_model.predict(input_for_prediction)
                        input_file.fillna('',inplace=True)
                        if request.form.get("Predict_View")=='Predict & View':
                            explainer=ClassifierExplainer(prediction_model, input_file[prediction_features],input_file['primary_close_flag'] ,idxs=input_file.index)
                            db=ExplainerDashboard(explainer,title='Default Prediction',model_summary=False,shap_dependence=False,shap_interaction=False,contribution=False)
                            db.run(host='http://localhost',port=5000)
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

@app.route('/file_upload_error',methods=['GET','POST'])
def user_file_upload_error():
    if request.method=='GET':
        return render_template('file_upload_error.html')
    else:
        if request.form.get("Upload_File")=='Go back to File Upload':
            return redirect(url_for('index'))            
        else:
            return None

if __name__=="__main__":
    app.run(debug=True)