from flask import Flask, request, jsonify
import pickle
import numpy as np
import sklearn

model = pickle.load(open('heart_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "hello world"

@app.route('/predict',methods =['POST'])
def predict():
#     age	sex	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal
    age = request.form['age']
    sex= request.form['sex']
    cp= request.form['cp']
    trestbps= request.form['trestbps']
    chol= request.form['chol']
    fbs= request.form['fbs']
    restecg= request.form['restecg']
    thalach= request.form['thalach']
    exang= request.form['exang']
    oldpeak= request.form['oldpeak']
    slope= request.form['slope']
    ca= request.form['ca']
    thal= request.form['thal']

    # result = {'age':age,'sex':	sex, 'cp':cp,'trestbps':trestbps,'chol':chol,'fbs':fbs,'restecg':restecg,'thalach':	thalach,'exang':exang,'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal}
    # result = {'age':age}
    input_query = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]], dtype=float)
    result = str(model.predict(input_query)[0])
    print(age)
    return jsonify({'Prediction':str(result)})



if __name__ == '__main__':
    app.run(debug=True)

