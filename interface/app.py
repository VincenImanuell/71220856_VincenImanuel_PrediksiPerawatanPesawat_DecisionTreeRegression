from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

# Load dataset
dt = pd.read_csv("data-train-pesawat.csv")
# Hapus kolom yang tidak digunakan
target_hapus = ['setting3', 's1', 's5', 's6', 's10', 's16', 's18', 's19', 'av1', 'av5', 'av6', 'av10', 'av16', 'av18', 'av19','sd1', 'sd5', 'sd6', 'sd10', 'sd16', 'sd18', 'sd19']
dt = dt.drop(columns=target_hapus)
# Hapus baris pertama yang berisi semua nilai 0
dt = dt.drop(index=0)

# Buat model Decision Tree Regression
x = dt[['s4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 'av2', 'av3', 'av4', 'av7', 'av8', 'av9', 'av11', 'av12', 'av13', 'av14', 'av15', 'av17', 'av20', 'av21']]
y_bnc = dt['label_bnc']
y_mcc = dt['label_mcc']
model_bnc = DecisionTreeRegressor()
model_bnc.fit(x, y_bnc)
model_mcc = DecisionTreeRegressor()
model_mcc.fit(x, y_mcc)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ambil nilai input dari form
        s4 = float(request.form['s4'])
        s7 = float(request.form['s7'])
        s8 = float(request.form['s8'])
        s9 = float(request.form['s9'])
        s11 = float(request.form['s11'])
        s12 = float(request.form['s12'])
        s13 = float(request.form['s13'])
        s14 = float(request.form['s14'])
        av2 = float(request.form['av2'])
        av3 = float(request.form['av3'])
        av4 = float(request.form['av4'])
        av7 = float(request.form['av7'])
        av8 = float(request.form['av8'])
        av9 = float(request.form['av9'])
        av11 = float(request.form['av11'])
        av12 = float(request.form['av12'])
        av13 = float(request.form['av13'])
        av14 = float(request.form['av14'])
        av15 = float(request.form['av15'])
        av17 = float(request.form['av17'])
        av20 = float(request.form['av20'])
        av21 = float(request.form['av21'])

        # Lakukan prediksi
        fitur = np.array([[s4, s7, s8, s9, s11, s12, s13, s14, av2, av3, av4, av7, av8, av9, av11, av12, av13, av14, av15, av17, av20, av21]])
        hasil_bnc = model_bnc.predict(fitur)
        hasil_mcc = model_mcc.predict(fitur)

        return render_template('result.html', hasil_bnc=int(hasil_bnc[0]), hasil_mcc=int(hasil_mcc[0]))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
