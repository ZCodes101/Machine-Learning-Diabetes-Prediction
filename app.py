from flask import Flask, render_template,request
import pickle
import numpy as np


model = pickle.load(open('diabetes_pred_model_2.pickle', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h'] 


    a = [[data1, data2, data3, data4,data5,data6,data7,data8]]
    arr = np.array(a,dtype=float)
    pred = model.predict(arr)
    return render_template('after.html', data=pred)
   

    

if __name__ == '__main__':
    app.run(port=3000, debug=True)
