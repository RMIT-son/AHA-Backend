# AHA-Backend
Backend for AHA system instruction for testing
1. Clone repo, cd to repo and run **python -m venv venv** to create virtual enviroment, run **venv\Scripts\activate** to activate
2. Run **pip install -r requirements.txt** in virtual enviroment to install all the required packages 
3. Create a **python file** copy and run the code below to install all the required models:
4. Run command **uvicorn app.main:app** to run FastAPI
5. Go to **localhost:8000/docs#/** to test the model

**Note:** check last 1 line in the requirements.txt if you cannot install requirements.txt, you will have to install that last 1 package separately so just command that 1 line temporary and run pip install -r requirements.txt again if it fails to install, and then run pip install that 1 package later
