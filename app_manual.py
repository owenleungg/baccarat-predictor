import streamlit as st
from typing import List
from baccarat_inference import BaccaratPredictor  # or your module path

# lazy-load model
@st.cache_resource
def load_predictor():
    return BaccaratPredictor(model_path="models/final_model.pth")

predictor = load_predictor()

st.title("Baccarat Predictor")
st.caption("Enter outcomes as P B T. Example: P B B T P")

# session state for running history
if "history" not in st.session_state:
    st.session_state.history: List[str] = []

col1, col2 = st.columns(2)
with col1:
    user_line = st.text_input("Add outcomes (space separated)", "")
with col2:
    file = st.file_uploader("Load outcomes file (.csv .txt .json)", type=["csv","txt","json"])

def clean_tokens(tokens):
    return [t for t in [x.strip().upper() for x in tokens] if t in {"P","B","T"}]

# handle text input add
if st.button("Add"):
    toks = clean_tokens(user_line.split())
    if toks:
        st.session_state.history.extend(toks)
        st.success(f"Added {len(toks)} outcomes")
    else:
        st.warning("No valid outcomes found. Use P, B, or T.")

# handle file upload
if file is not None and st.button("Load file"):
    import pandas as pd, json
    outcomes = []
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file)
            for col in ["outcome","Outcome","result","Result"]:
                if col in df.columns:
                    outcomes = df[col].dropna().astype(str).str.upper().tolist()
                    break
            if not outcomes and df.shape[1] >= 1:
                outcomes = df.iloc[:,0].dropna().astype(str).str.upper().tolist()
        elif name.endswith(".txt"):
            text = file.read().decode("utf-8")
            sep = "," if "," in text else " "
            outcomes = [t.strip().upper() for t in text.split(sep)]
        else:  # json
            data = json.load(file)
            if isinstance(data, list):
                outcomes = [str(x).upper() for x in data]
            elif isinstance(data, dict) and "outcomes" in data:
                outcomes = [str(x).upper() for x in data["outcomes"]]
        toks = clean_tokens(outcomes)
        st.session_state.history.extend(toks)
        st.success(f"Loaded {len(toks)} outcomes")
    except Exception as e:
        st.error(f"Load error: {e}")

# controls
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Clear"):
        st.session_state.history.clear()
with c2:
    if st.button("Undo last"):
        if st.session_state.history:
            st.session_state.history.pop()
with c3:
    predict_clicked = st.button("Predict")

st.subheader("Current history")
st.write(f"{len(st.session_state.history)} outcomes")
st.code(" ".join(st.session_state.history[-72:]) or "(empty)")

def bar(p):  # text bar
    n = int(p*20)
    return "█"*n + "░"*(20-n)

if predict_clicked and st.session_state.history:
    try:
        result = predictor.predict_with_analysis(st.session_state.history)
        pred = result["prediction"]
        st.subheader("Prediction")
        st.write(f"Next: **{pred['predicted_outcome']}**")
        st.write(f"Confidence: **{pred['confidence']:.1%}**  "
                 f"({result['game_state']['confidence_level']})")
        st.write(f"Data quality: {result['game_state']['data_sufficiency']}")

        if "probabilities" in pred:
            st.subheader("Probabilities")
            for k,v in pred["probabilities"].items():
                st.write(f"{k}: {v:.1%}  {bar(v)}")

        st.subheader("Pattern")
        pat = result["pattern_analysis"]
        st.write(f"Current streak: {pat['current_streak']['outcome']} × {pat['current_streak']['length']}")
        st.write(f"Last 5: {pat['recent_distribution']['last_5_games']}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
