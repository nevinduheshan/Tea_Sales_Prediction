from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and encoder
model = pickle.load(open(r'C:\\Users\\nevin\\OneDrive\\Desktop\\ICBT Bsc\\final assignment\\final proj\\my_new\\trained_model.pickle', 'rb'))
encoder = pickle.load(open(r'C:\\Users\\nevin\\OneDrive\\Desktop\\ICBT Bsc\\final assignment\\final proj\\my_new\\encoder.pickle', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    
    # Preprocessing similar to your FastAPI implementation
    df.rename(columns={
        'date': 'Date',
        "selling_mark": "Selling Mark",
        "grade": "Grade",
        "invoice_no": 'Invoice No',
        "lot_no": 'Lot No',
        "bag_weight": 'Bag Weight',
        "no_of_bags": 'No of Bags'
    }, inplace=True)
    
    df[["day", "month", "year"]] = df["Date"].str.split("/", expand=True)
    df.drop(columns=["Date"], inplace=True)
    df = df.astype({"day": int, "month": int, "year": int})
    
    df_objects = df.loc[:, ["Selling Mark", "Grade"]]
    df_objects_t = encoder.transform(df_objects).toarray()
    
    df.drop(columns=["Selling Mark", "Grade"], inplace=True)
    enc_list = encoder.categories_[0].tolist() + encoder.categories_[1].tolist()
    df_t = pd.DataFrame(df_objects_t, columns=enc_list)
    
    dff = pd.concat([df, df_t], axis=1)
    
    # Convert 'no_of_bags' and 'bag_weight' to numeric types
    no_of_bags = int(data['no_of_bags'])
    bag_weight = float(data['bag_weight'])
    
    # Prediction
    result = model.predict(dff)
    
    # Calculate nett_qty and prepare the response
    nett_qty = no_of_bags * bag_weight
    response = {"price": round(result[0], 2), "amount": round(result[0] * nett_qty, 2)}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
