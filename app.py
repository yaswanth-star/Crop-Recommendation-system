from flask import Flask,render_template,request,session,jsonify,redirect
import sqlite3
from datetime import datetime
import pickle
import joblib
import numpy as np
import pandas as pd
import webbrowser
import requests
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("VERCEL"):
    DB_PATH = "/tmp/cropwise.db"
    if not os.path.exists(DB_PATH):
        shutil.copyfile(os.path.join(BASE_DIR, "cropwise.db"), DB_PATH)
else:
    DB_PATH = os.path.join(BASE_DIR, "cropwise.db")

app=Flask(__name__)
app.secret_key="secret"

def init_db():
    conn=sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            crop TEXT NOT NULL,
            confidence REAL NOT NULL,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            ph REAL,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            city TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

model=pickle.load(open("model.pkl","rb"))
features=pickle.load(open("features.pkl","rb"))
p_model=joblib.load("p_model.pkl")
k_model=joblib.load("k_model.pkl")

@app.route('/')
def login():
    return render_template("login.html")

@app.route('/login',methods=['POST'])
def login_user():
    email=request.form.get('email','').strip()
    password=request.form.get('password','').strip()
    if email.endswith('@gmail.com') and password:
        session['email']=email
        return redirect('/dashboard')
    return "Please enter a valid Gmail address and password."

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',active_page='dashboard')

@app.route('/history')
def history():
    if 'email' not in session:
        return redirect('/')
    conn=sqlite3.connect("cropwise.db")
    conn.row_factory=sqlite3.Row
    records=conn.execute(
        "SELECT * FROM history WHERE email=? ORDER BY id DESC",
        (session['email'],)
    ).fetchall()
    conn.close()
    return render_template("history.html",records=records,active_page='history')

@app.route('/profile')
def profile():
    if 'email' not in session:
        return redirect('/')
    email=session['email']
    initial=email[0].upper() if email else 'U'
    name=email.split('@')[0].replace('.',' ').replace('_',' ').title()
    conn=sqlite3.connect("cropwise.db")
    total=conn.execute(
        "SELECT COUNT(*) FROM history WHERE email=?",
        (email,)
    ).fetchone()[0]
    last=conn.execute(
        "SELECT crop,created_at FROM history WHERE email=? ORDER BY id DESC LIMIT 1",
        (email,)
    ).fetchone()
    conn.close()
    return render_template(
        "profile.html",
        email=email,
        initial=initial,
        name=name,
        total=total,
        last=last,
        active_page='profile'
    )

@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/')

@app.route('/soil')
def soil():
    return render_template("soil.html",active_page='soil')

@app.route('/climate',methods=['POST'])
def climate():
    city=request.form.get('city','').strip()
    session['city']=city
    session['P']=request.form.get('P',40)
    session['K']=request.form.get('K',35)
    session['nitrogen']=request.form.get('nitrogen',1.4)
    session['ph']=request.form.get('ph',6.5)
    session['clay']=request.form.get('clay',25.0)
    session['organic_carbon']=request.form.get('organic_carbon',12.0)
    session['latitude']=request.form.get('latitude',17.3850)
    session['longitude']=request.form.get('longitude',78.4867)
    return render_template("climate.html",locked_city=city)

@app.route('/get_soil_by_city',methods=['POST'])
def get_soil_by_city():
    data=request.get_json() or {}
    city=data.get('city','').strip()
    if not city:
        return jsonify({
            "success":False,
            "message":"City name is required."
        }),400
    try:
        geo_response=requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name":city,
                "count":1,
                "language":"en",
                "format":"json"
            },
            timeout=10
        )
        if geo_response.status_code!=200:
            print("GEOCODING STATUS:",geo_response.status_code)
            print("GEOCODING RESPONSE:",geo_response.text[:500])
            return jsonify({
                "success":False,
                "message":"Unable to find city coordinates."
            }),500
        try:
            geo=geo_response.json()
        except ValueError:
            print("INVALID GEOCODING RESPONSE:",geo_response.text[:500])
            return jsonify({
                "success":False,
                "message":"City API returned invalid data."
            }),500
        results=geo.get("results")
        if not results:
            return jsonify({
                "success":False,
                "message":f"City '{city}' not found."
            }),404
        c=results[0]
        lat=c["latitude"]
        lon=c["longitude"]
        soil_url="https://rest.isric.org/soilgrids/v2.0/properties/query"
        soil_params=[
            ("lat",lat),
            ("lon",lon),
            ("property","nitrogen"),
            ("property","phh2o"),
            ("property","clay"),
            ("property","soc"),
            ("depth","0-5cm"),
            ("value","mean")
        ]
        soil_response=requests.get(
            soil_url,
            params=soil_params,
            timeout=30
        )
        print("\n==============================")
        print("SOILGRIDS STATUS:",soil_response.status_code)
        print("SOILGRIDS RESPONSE:",soil_response.text[:500])
        print("==============================\n")
        if soil_response.status_code!=200:
            return jsonify({
                "success":False,
                "message":"SoilGrids API is currently unavailable. Please try again later."
            }),503
        if not soil_response.text.strip():
            return jsonify({
                "success":False,
                "message":"SoilGrids returned an empty response."
            }),503
        try:
            soil=soil_response.json()
        except ValueError:
            print("INVALID SOILGRIDS RESPONSE:")
            print(soil_response.text[:500])
            return jsonify({
                "success":False,
                "message":"SoilGrids returned an invalid response."
            }),503
        layers={
            x["name"]:x["depths"][0]["values"]["mean"]
            for x in soil.get("properties",{}).get("layers",[])
        }
        nitrogen=round((layers.get("nitrogen") or 140)/100.0,2)
        ph=round((layers.get("phh2o") or 65)/10.0,1)
        clay=round((layers.get("clay") or 250)/10.0,1)
        organic_carbon=round((layers.get("soc") or 120)/10.0,1)
        session['city']=c.get("name",city)
        session['latitude']=lat
        session['longitude']=lon
        session['nitrogen']=nitrogen
        session['ph']=ph
        session['clay']=clay
        session['organic_carbon']=organic_carbon
        return jsonify({
            "success":True,
            "city":c.get("name",city),
            "country":c.get("country",""),
            "latitude":lat,
            "longitude":lon,
            "nitrogen":nitrogen,
            "ph":ph,
            "clay":clay,
            "organic_carbon":organic_carbon
        })
    except requests.exceptions.Timeout:
        return jsonify({
            "success":False,
            "message":"Soil API request timed out. Please try again."
        }),504
    except requests.exceptions.RequestException as e:
        print("REQUEST ERROR:",e)
        return jsonify({
            "success":False,
            "message":"Unable to connect to the soil API."
        }),503
    except Exception as e:
        print("GENERAL API ERROR:",e)
        return jsonify({
            "success":False,
            "message":f"API Error: {str(e)}"
        }),500

@app.route('/get_weather_by_city',methods=['POST'])
def get_weather_by_city():
    data=request.get_json() or {}
    city=data.get('city','').strip()
    days=int(data.get('range',30))
    session_city=session.get('city','').strip()
    if session_city and city.lower()!=session_city.lower():
        return jsonify({
            "success":False,
            "message":f"City mismatch! You selected '{session_city}' on the soil page."
        }),400
    if not city:
        return jsonify({
            "success":False,
            "message":"Please enter a city name."
        }),400
    try:
        geo_response=requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name":city,"count":1},
            timeout=5
        )
        geo=geo_response.json()
        results=geo.get("results")
        if not results:
            return jsonify({
                "success":False,
                "message":f"City '{city}' not found."
            }),404
        c=results[0]
        past=max(0,days-1)
        weather_response=requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":c["latitude"],
                "longitude":c["longitude"],
                "current":"relative_humidity_2m",
                "daily":"temperature_2m_mean,precipitation_sum",
                "past_days":past,
                "forecast_days":1,
                "timezone":"auto"
            },
            timeout=5
        )
        weather=weather_response.json()
        daily=weather.get("daily",{})
        temps=[
            x for x in daily.get("temperature_2m_mean",[])
            if x is not None
        ]
        rain=[
            x for x in daily.get("precipitation_sum",[])
            if x is not None
        ]
        temp=round(float(np.mean(temps)),2) if temps else 25.0
        rainfall=round(max(float(np.sum(rain))*3,50),2) if rain else 50.0
        humidity=weather.get("current",{}).get(
            "relative_humidity_2m",
            70.0
        )
        nitrogen=float(session.get('nitrogen',1.4))
        ph=float(session.get('ph',6.5))
        clay=float(session.get('clay',25.0))
        organic_carbon=float(session.get('organic_carbon',12.0))
        pk_input=[[
            nitrogen,
            ph,
            clay,
            organic_carbon,
            temp,
            humidity,
            rainfall
        ]]
        predicted_p=round(float(p_model.predict(pk_input)[0]),2)
        predicted_k=round(float(k_model.predict(pk_input)[0]),2)
        session['P']=predicted_p
        session['K']=predicted_k
        session['temperature']=temp
        session['humidity']=humidity
        session['rainfall']=rainfall
        print("\nP:",predicted_p)
        print("K:",predicted_k)
        return jsonify({
            "success":True,
            "city":c.get("name",""),
            "country":c.get("country",""),
            "temperature":temp,
            "humidity":humidity,
            "rainfall":rainfall,
            "phosphorus":predicted_p,
            "potassium":predicted_k
        })
    except Exception as e:
        return jsonify({
            "success":False,
            "message":str(e)
        }),500

@app.route('/predict',methods=['POST'])
def predict():
    record={
        'P':float(session.get('P',40)),
        'K':float(session.get('K',35)),
        'nitrogen':float(session.get('nitrogen',1.4)),
        'ph':float(session.get('ph',6.5)),
        'clay':float(session.get('clay',25.0)),
        'organic_carbon':float(session.get('organic_carbon',12.0)),
        'temperature':float(request.form['temperature']),
        'humidity':float(request.form['humidity']),
        'rainfall':float(request.form['rainfall'])
    }
    input_df=pd.DataFrame([record])[features]
    probs=model.predict_proba(input_df)[0]
    classes=model.classes_
    idx=probs.argsort()[-3:][::-1]
    print("\nINPUT TO MODEL:")
    print(input_df)
    print("\nPREDICTIONS:")
    for i in idx:
        print(classes[i],round(probs[i]*100,2))
    top3=[
        (classes[i],round(probs[i]*100,2))
        for i in idx
    ]
    conn=sqlite3.connect("cropwise.db")
    conn.execute("""
        INSERT INTO history(
            email,
            crop,
            confidence,
            nitrogen,
            phosphorus,
            potassium,
            ph,
            temperature,
            humidity,
            rainfall,
            city,
            created_at
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
    """,(
        session.get("email",""),
        top3[0][0],
        top3[0][1],
        record["nitrogen"],
        record["P"],
        record["K"],
        record["ph"],
        record["temperature"],
        record["humidity"],
        record["rainfall"],
        session.get("city",""),
        datetime.now().strftime("%d-%m-%Y %I:%M %p")
    ))
    conn.commit()
    conn.close()
    soil_data={
        "N":record["nitrogen"],
        "P":record["P"],
        "K":record["K"],
        "ph":record["ph"],
        "temperature":record["temperature"],
        "humidity":record["humidity"],
        "rainfall":record["rainfall"]
    }
    return render_template(
        "result.html",
        top3=top3,
        soil_data=soil_data,
        active_page='results'
    )

if __name__=="__main__":
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)