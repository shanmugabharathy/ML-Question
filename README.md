# Solution
The entire data generated and processed during the submission is stored here: https://drive.google.com/drive/folders/1veQ0hCBBLCDdB56HbuonPmqrMqYCgigd?usp=sharing

## File Structure
- `preprocess.ipynb`: This is the script for taking the raw `news_articles.json` data and process into a proper dataset.
- `data_ingestion.ioynb`: This is the script to perform data ingestion on the processed data and store (split, chunk and index) into the ChromaDB.
- `SOLUTION.ipynb`: This file has the entire code for data ingestion along with the complete Question Answering system.
- `script.py`: It is a standalone python script that can answer a single question at a time.
- `main.py`: The streamlit application for the Question Answering System.
- `requirements.txt`: All the dependencies required to run the code locally.

## Steps to setup locally
- Clone the repository
```
git clone git@github.com:Sakalya100/ML-Question.git
```

- Go to the directory
```
cd ML-Question
```

- Create a virtual Environment
```
python -m venv myenv
```

- Install the dependencies
```
pip install -r requirements.txt
```

- Download the processed data or download the rawe data and run the preprocessing script.
- With the processed data, exectue the `data_ingestion.ipynb`.

- Set your Environment Variables with your own HuggingFace API Key and Chroma Path
```
DATA_PATH = "data"
CHROMA_PATH = "chroma"
HUGGINGFACEHUB_API_TOKEN = "your_api_key"
```

- Run the streamlit application
```
streamlit run main.py
```

Head to the `http://localhost:8501` to see your app running.
