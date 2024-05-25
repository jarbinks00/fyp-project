# save this as app.py
from flask import Flask, request, render_template, url_for
import pickle
import numpy as np
import pandas as pd
import json
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__,template_folder='templates', static_folder='static')
model = pickle.load(open('D:/FYP_System_LaptopPricePrediction/Web Application Development/RandomForestRegressorModel.pkl', 'rb'))
data = pd.read_csv(r'D:/FYP_System_LaptopPricePrediction/Web Application Development/Cleaned_Laptop_Dataset.csv')

@app.route('/')
def home_dashboard():
    

    # Graph One Pie Chart
    new = pd.DataFrame({
        'Brand': data['Company'].value_counts().index,
        'Value': data['Company'].value_counts()
    })
    fig1 = px.pie(new, values='Value', names='Brand')
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    # Graph two Bar Chart Vertical
    final_df = pd.DataFrame()
    X_agg = data.groupby('Company', as_index=False).agg({'Price_MYR': ['count', 'sum', 'mean']})
    X_agg.columns = ['Brand', 'sale_count', 'selling_sum', 'Average Price']
    final_df = pd.concat([final_df, X_agg])
    final_df = final_df.groupby('Brand', as_index=False).agg({'sale_count': 'sum', 'selling_sum': 'sum', 'Average Price': 'sum'})
    fig2 = px.bar(final_df, x='Brand', y='Average Price', color='Brand')
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # Graph three Bar Chart Horizontal
    dftype = pd.DataFrame()
    typedf = data.groupby('TypeName', as_index=False).agg({'Price_MYR': ['count', 'sum', 'mean']})
    typedf.columns = ['Type of Laptop', 'sale_count', 'selling_sum', 'Average Price']
    dftype = pd.concat([dftype, typedf])
    dftype = dftype.groupby('Type of Laptop', as_index=False).agg({'sale_count': 'sum', 'selling_sum': 'sum', 'Average Price': 'sum'})
    fig3 = px.bar(dftype, x="Average Price", y="Type of Laptop", color='Type of Laptop')
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    # Graph four Doughnout Chart
    df_four = pd.DataFrame()
    four_df = data.groupby('Operating System', as_index=False).agg({'Price_MYR': ['count', 'sum', 'mean']})
    four_df.columns = ['Type of Operating System', 'sale_count', 'selling_sum', 'Average Price']
    df_four = pd.concat([df_four, four_df])
    df_four = df_four.groupby('Type of Operating System', as_index=False).agg({'sale_count': 'sum', 'selling_sum': 'sum', 'Average Price': 'sum'})
    fig4 = px.pie(df_four, values="Average Price", names="Type of Operating System", color='Type of Operating System', hole=.3)
    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    # Graph five Bar Chart Vertical
    df_fifth = pd.DataFrame()
    fifth_df = data.groupby('GPU Brand', as_index=False).agg({'Price_MYR': ['count', 'sum', 'mean']})
    fifth_df.columns = ['Graphical Processing Unit Brand', 'sale_count', 'selling_sum', 'Average Price']
    df_fifth = pd.concat([df_fifth, fifth_df])
    df_fifth = df_fifth.groupby('Graphical Processing Unit Brand', as_index=False).agg({'sale_count': 'sum', 'selling_sum': 'sum', 'Average Price': 'sum'})
    fig5 = px.bar(df_fifth, x="Graphical Processing Unit Brand", y="Average Price", color='Graphical Processing Unit Brand')
    graph5JSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

    # Graph six Bar Chart Horizontal
    df_sixth = pd.DataFrame()
    sixth_df = data.groupby('Cpu brand', as_index=False).agg({'Price_MYR': ['count', 'sum', 'mean']})
    sixth_df.columns = ['Central Processing Unit Brand', 'sale_count', 'selling_sum', 'Average Price']
    df_sixth = pd.concat([df_sixth, sixth_df])
    df_sixth = df_sixth.groupby('Central Processing Unit Brand', as_index=False).agg({'sale_count': 'sum', 'selling_sum': 'sum', 'Average Price': 'sum'})
    fig6 = px.bar(df_sixth, x="Average Price", y="Central Processing Unit Brand", color='Central Processing Unit Brand')
    graph6JSON = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)

    # # Use `hole` to create a donut-like pie chart
    # fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    # fig.show()

    return render_template("dashboard.html", graph1JSON=graph1JSON, graph2JSON=graph2JSON, graph3JSON=graph3JSON, graph4JSON=graph4JSON, graph5JSON=graph5JSON, graph6JSON=graph6JSON)

#DataType Which are STRING VALUE
brand = sorted(data['Company'].unique())
type = sorted(data['TypeName'].unique())
cpu = sorted(data['Cpu brand'].unique())
gpu = sorted(data['GPU Brand'].unique())
os = sorted(data['Operating System'].unique())

@app.route('/form_page')
def form_page():
    return render_template("form.html", brand=brand, type=type, cpu=cpu, gpu=gpu, os=os)

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        brand = request.form['brand']
        type = request.form['type']
        RAM = request.form['RAM']
        weight = request.form['weight']
        touchscreen = request.form['touchscreen']
        IPS = request.form['IPS']
        CPU_GHz = float(request.form['CPU_GHz'])
        screen_size = float(request.form['screen_size'])
        screen_resolution = request.form['screen_resolution']
        CPU = request.form['CPU']
        HDD = request.form['HDD']
        SSD = request.form['SSD']
        GPU = request.form['GPU']
        OS = request.form['OS']

        ppi = None
        if touchscreen == 'Yes':
            Touchscreen = 1
        else:
            Touchscreen = 0

        if IPS == 'Yes':
            ips = 1
        else:
            ips = 0

        X_res = int(screen_resolution.split('x')[0])
        Y_res = int(screen_resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # print(brand,type,RAM,weight,Touchscreen,IPS,ppi, CPU_GHz,CPU,HDD,SSD,GPU,OS)


        query = np.array([brand,type,RAM,weight,Touchscreen,ips,ppi,CPU_GHz,CPU,HDD,SSD,GPU,OS])
        query = query.reshape(1, 13)
        prediction = str(int(np.exp(model.predict(query)[0])))
        return render_template("predict.html", prediction_text="The Predicted Laptop Price of This Configuration is RM{}".format(prediction), brand=brand, type=type, RAM=RAM, weight=weight, touchscreen=touchscreen, IPS=IPS, CPU_GHz=CPU_GHz, screen_size=screen_size, screen_resolution=screen_resolution, CPU=CPU, HDD=HDD, SSD=SSD, GPU=GPU, OS=OS)

if __name__ == "__main__":
    app.run(debug=True)