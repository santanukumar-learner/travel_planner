# Rourkela Travel Planner

Run the local Streamlit planner from this folder:

```powershell
python -m pip install -r requirements-streamlit.txt
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501). Select a date, starting place, radius, hours, and optional stops. The app creates a route, shows the stops on a map, and can export the itinerary as JSON. No coordinates need to be entered.

If Windows shows the Microsoft Store "Python was not found" message in an older VS Code terminal, open a new terminal or run `& "$env:LOCALAPPDATA\Python\bin\python.exe" -m streamlit run streamlit_app.py`. You can also use `powershell -ExecutionPolicy Bypass -File .\run_streamlit.ps1` from this folder.

This local version uses `rag.json` and needs no API keys, Docker, or Pathway. Travel duration is an estimate based on 20 km/h with one hour at each stop. Places without coordinates cannot be routed. The older Gemini, weather, and crowd prediction modules remain separate and are not used by this Streamlit app.

## Render

`render.yaml` defines a free Python web service. Put this folder's app files in the root of a Git repository, connect the repository to Render, and create a Blueprint from it. Render will install `requirements-streamlit.txt`, start Streamlit on its assigned port, and check `/_stcore/health`. The old `Requirements.txt` and Docker files are not used by this service.
