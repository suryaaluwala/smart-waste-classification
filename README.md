# Smart Waste Classification

Deep learning project that classifies waste images into the six TrashNet categories and provides cautious disposal guidance in a Streamlit app.

Project source, setup, and training notes: [Smart_Waste_Classification/README.md](Smart_Waste_Classification/README.md)

## Run locally

```powershell
cd Smart_Waste_Classification
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Connect this repository to Streamlit Community Cloud and use:

- Branch: `master`
- Main file path: `Smart_Waste_Classification/app.py`
- Dependency file: `Smart_Waste_Classification/requirements.txt`

The trained baseline checkpoint is included in `Smart_Waste_Classification/models/`. The image dataset and local virtual environments are excluded from Git.
