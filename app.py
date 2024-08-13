from fastapi import FastAPI
import pickle

from note import BankNote

app = FastAPI()



@app.get('/')      # in flask instead of get we use route
def read_root():
    return {'message':"Hello, World !"}

@app.get("/item/{item_id}")
def read_item(item_id: int, q:str = None):
    return {"item_id":item_id, "q":q}


@app.post('/predict')
def predict(data: BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    with open('model.pkl', 'rb') as picle_in:
        model = pickle.load(picle_in)
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    
    if prediction[0] == 0:
        pred = 'The note is fake'
    else:
        pred = 'The note is original'
    return {'prediction': pred}