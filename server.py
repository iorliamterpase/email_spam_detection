from flask import Flask, render_template, request
import pickle

# Load tokenizer and model
tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form.get('content')  
    tokenized_email = tokenizer.transform([email_text])  
    prediction = model.predict(tokenized_email)[0]  
    prediction = 1 if prediction == 1 else -1  
    return render_template('home.html', prediction=prediction, email_text=email_text)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
