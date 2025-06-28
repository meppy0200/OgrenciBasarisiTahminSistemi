from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Modeli yükle
print("Model yükleniyor...")
model_data = joblib.load("model/student_model.pkl")
model = model_data["model"]
features = model_data["features"]
feature_importance = model_data["feature_importance"]

print(f"Model yüklendi. Özellik sayısı: {len(features)}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/explore")
def explore():
    df = pd.read_csv("data/StudentsPerformance.csv")
    
    stats = {
        'total_students': len(df),
        'avg_math': round(df["math score"].mean(), 2),
        'avg_reading': round(df["reading score"].mean(), 2),
        'avg_writing': round(df["writing score"].mean(), 2),
        'gender_dist': df["gender"].value_counts().to_dict(),
        'race_dist': df["race/ethnicity"].value_counts().to_dict(),
        'education_dist': df["parental level of education"].value_counts().to_dict(),
        'lunch_dist': df["lunch"].value_counts().to_dict(),
        'prep_dist': df["test preparation course"].value_counts().to_dict()
    }
    
    df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    success_rate = round((df["average_score"] >= 60).mean() * 100, 2)
    stats['success_rate'] = success_rate
    
    return render_template("explore.html", stats=stats)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    probability = None
    
    if request.method == "POST":
        gender = request.form["gender"]
        race = request.form["race"]
        parental_ed = request.form["parental_education"]
        lunch = request.form["lunch"]
        test_prep = request.form["test_prep"]
        
        # DataFrame
        input_data = {
            "gender": gender,
            "race/ethnicity": race,
            "parental level of education": parental_ed,
            "lunch": lunch,
            "test preparation course": test_prep
        }
        
        input_df = pd.DataFrame([input_data])
        
        input_df = pd.get_dummies(input_df)
        
        for col in features:
            if col not in input_df:
                input_df[col] = 0
        
        input_df = input_df[features]
        
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        
        result = "🎉 Başarılı" if prediction == 1 else "❌ Başarısız"
        probability = round(prob[1] * 100, 2)  # Başarı olasılığı
    
    return render_template("predict.html", result=result, probability=probability)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        
        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df)
        
        for col in features:
            if col not in input_df:
                input_df[col] = 0
        
        input_df = input_df[features]
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': round(probability * 100, 2),
            'result': "Başarılı" if prediction == 1 else "Başarısız"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route("/feature_importance")
def feature_importance_page():
    """Özellik önemlerini göster"""
    return render_template("feature_importance.html", 
                         features=feature_importance.head(15).to_dict('records'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)