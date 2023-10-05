from flask import Flask,render_template
from flask import request
import pandas as pd
import  numpy as np
import pickle



df1 = pd.read_csv('appartments1.csv')
df2 = df1.set_index('PropertyName')
location_df = pd.read_csv('location_df1.csv')
with open('cosine.pkl', 'rb') as file:
    cosine = pickle.load(file)

with open('df.pkl','rb') as file:
    df = pickle.load(file)
    df[['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']]=df[['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']].astype(int)

with open('pipeline.pkl','rb') as file:
     pipeline = pickle.load(file)

app =  Flask(__name__)

@app.route('/Analysis')
def ana():
    return render_template('analysis.html')

@app.route('/recommender')
def home1():
    return render_template('recommender.html',
                           landmark=list(location_df.columns.values)
                           )

@app.route('/recommend_building',methods=['POST'])
def home2():
    selected_landmark = request.form.get('user_input[]')
    selected_kms = request.form.get('kms')
    selected_kms = float(selected_kms) * 1000
    #selected_kms = [int(km) for km in selected_kms]
    print(selected_landmark)
    dd = location_df[location_df[selected_landmark] < selected_kms+1]['PropertyName'].values
    df2 = df1[df1.PropertyName.isin(dd)]



    # dd = location_df[location_df[selected_landmark] < selected_kms+1].index
    # df2=df1[df1.PropertyName.isin(dd)]
    return render_template('recommender.html',
                           landmark=list(location_df.columns.values),
                           img_url=list(df2['img'].values),
                           property_name = list(df2['PropertyName'].values),
                           sector2=list(df2['PropertySubName'].values),
                           facility=list(df2['TopFacilities'].values),
                           price_low=list(df2['price low'].values),
                           price_high=list(df2['price high'].values))


@app.route('/prediction')
def index1():
    return render_template('prediction.html',
                           sector=sorted(df['sector'].unique().tolist()),
                           property=sorted(df['property_type'].unique().tolist()),
                           bedroom=sorted(df['bedRoom'].unique().tolist()),
                           bathroom=sorted(df['bathroom'].unique().tolist()),
                           balcony=sorted(df['balcony'].unique().tolist()),
                           age=sorted(df['agePossession'].unique().tolist()),
                           servantroom=sorted(df['servant room'].unique().tolist()),
                           storeroom=sorted(df['store room'].unique().tolist()),
                           furnishing=sorted(df['furnishing_type'].unique().tolist()),
                           floor=sorted(df['floor_category'].unique().tolist()))



@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/home')
def index2():
    return render_template('Home.html')

@app.route('/predict_price', methods=['POST'])
def recommend_food1():
    property1 = request.form.get('property')
    sector1 = request.form.get('sector')
    bedroom1 = request.form.get('bedroom')
    bathroom1 = request.form.get('bathroom')
    balcony1 = request.form.get('balcony')
    age1 = request.form.get('age')
    servantroom1 = request.form.get('servantroom')
    storeroom1 = request.form.get('storeroom')
    furnishing1 = request.form.get('furnishing')
    floor1 = request.form.get('floor')
    built_up1 = request.form.get('built_up')

    data = [[property1, sector1, bedroom1, bathroom1, balcony1, age1, built_up1, servantroom1, storeroom1,
             furnishing1, floor1]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'floor_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    # predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    base_price = float(base_price)
    low = round(base_price - 0.20,2)
    high = round(base_price + 0.20,2)
    return render_template('prediction.html',
                           sector=sorted(df['sector'].unique().tolist()),
                           property=sorted(df['property_type'].unique().tolist()),
                           bedroom=sorted(df['bedRoom'].unique().tolist()),
                           bathroom=sorted(df['bathroom'].unique().tolist()),
                           balcony=sorted(df['balcony'].unique().tolist()),
                           age=sorted(df['agePossession'].unique().tolist()),
                           servantroom=sorted(df['servant room'].unique().tolist()),
                           storeroom=sorted(df['store room'].unique().tolist()),
                           furnishing=sorted(df['furnishing_type'].unique().tolist()),
                           floor=sorted(df['floor_category'].unique().tolist()),
                           low1=float(low),
                           high1=float(high))

@app.route('/perform_registration', methods=['POST'])
def perform_registration():
    nam = request.form.get('restroName')
    nam = str(nam)
    print(nam)
    index = np.where(df1.PropertyName == nam)[0][0]
    print(index)
    items = sorted(list(enumerate(cosine[index])), key=lambda x: x[1], reverse=True)[0:6]
    #return str(items)
    data = []

    for i in items:
        li = []
        temp_df = df1[df1.PropertyName == df2.index[i[0]]]
        li.extend(list(temp_df['img'].values))
        li.extend(list(temp_df['PropertyName'].values))
        li.extend(list(temp_df['PropertySubName'].values))
        li.extend(list(temp_df['TopFacilities'].values))
        li.extend(list(temp_df['price low'].values))
        li.extend(list(temp_df['price high'].values))
        # Add other data you want to collect here
        data.append(li)
    print(data)
    return render_template('rerecommender.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)