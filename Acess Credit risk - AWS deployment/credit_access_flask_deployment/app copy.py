import flask; print("flask version",flask.__version__)
from flask import Flask,render_template,request
import os
import numpy as np
import pickle

app = Flask(__name__)
app.env = "development"
result = ""
print("I am in flask app: ",app.env)

@app.route('/',methods=['GET','POST'])
def index():
    result = ""
    
    print('Request.method:',request.method)
    print('Request.TYPE',type(request))
    print('In the process of making a prediction')

    if request.method == 'POST':
        print("Request.form:",request.form)
        print(request.form['age'])
        print(request.form['sex'])
        print(request.form['job'])
        print(request.form['housing'])

        age = request.form['age']
        sex = request.form['sex']
        job = request.form['job']
        housing = request.form['housing']
        saving_account = request.form['saving_account']
        checking_amount = request.form['checking_amount']
        credit_amount = request.form['credit_amount']
        duration = request.form['duration']
        purpose = request.form['purpose']
        print(age,sex,job,housing,saving_account,checking_amount,credit_amount,duration,purpose)
        test_arr = np.array([age,sex,job,housing,saving_account,checking_amount,credit_amount,duration,purpose])
        print(test_arr)
        model = pickle.load(open('credit_risk_01_ml_model.pkl','rb'))
        print("Model Object: ",model)
        prediction = model.predict(test_arr)
        prediction = 'Risky' if prediction else "No Risk"
        result = f"The model has predicted that the result is: {prediction} "
        return render_template('result.html',result=result)
    return render_template('index.html',result=result)


@app.route('/result',methods=['GET'])
def result():
    # render the result page
    return render_template('result.html')

if __name__ == '__main__':
    app.run(host = 'localhost',port = 5001,debug = False)