# install libraries ---
#pip install fastapi uvicorn scikit-learn

# 1. Library imports
import uvicorn
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
import pickle

# 2. Create the app object
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. load the model
rgModel = pickle.load(open("rfr.pkl", "rb"))

# 4. Index route, opens automatically on http://127.0.0.1:80
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get("/predictAvg_Rainfall")
def getUserInfo(Year : int,Month : int):
    prediction = rgModel.predict([[Year,Month]])
    return {'Avg_Rainfall': prediction[0]}

# 5. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, port=80, host='0.0.0.0')
    
# uvicorn app:app --host 0.0.0.0 --port 80
# http://127.0.0.1/predictAvg_Rainfall?Year=2030&Month=11

