# app_one_tab.py
import streamlit as st
st.set_page_config(page_title="Baccarat Predictor", layout="centered")

import numpy as np, cv2, io
from typing import List
from PIL import ImageGrab, Image
import io


try:
    from streamlit_paste_image import paste_image
except Exception:
    paste_image = None

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.baccarat_inference import BaccaratPredictor

ROWS, COLS = 6, 24
ROW_PIX = 64
H_NORM = ROWS * ROW_PIX
def IS_RED(H,S,V):   return (S>=90)&(V>=80)&(((H<=12)|(H>=170))|((H>=15)&(H<=35)))
def IS_BLUE(H,S,V):  return (S>=90)&(V>=80)&(H>=95)&(H<=140)
def IS_GREEN(H,S,V): return (S>=90)&(V>=80)&(H>=60)&(H<=95)

@st.cache_resource
def load_predictor():
    return BaccaratPredictor(model_path="models/final_model.pth")
pred = load_predictor()

def find_board_roi(bgr):
    H,W = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g,(5,5),0)
    bw = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,5)
    edges = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),1)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best=None; score=-1.0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c); area=w*h
        if area < 0.2*H*W: continue
        asp = w/max(h,1)
        if asp < 2.2: continue
        rect = cv2.contourArea(c)/max(area,1)
        if rect < 0.70: continue
        s = area*asp*rect
        if s>score: score=s; best=(x,y,w,h)
    if best is None: return bgr
    x,y,w,h = best
    dx,dy = int(0.01*w), int(0.01*h)
    return bgr[y+dy:y+h-dy, x+dx:x+w-dx]

def rectify(bgr):
    scale = H_NORM / bgr.shape[0]
    W = int(bgr.shape[1]*scale)
    return cv2.resize(bgr, (W, H_NORM), cv2.INTER_AREA)

def split_cells(bgr):
    H,W,_ = bgr.shape
    rh = H//ROWS; cw = W//COLS
    for c in range(COLS):
        x0,x1 = c*cw, (c+1)*cw
        for r in range(ROWS):
            y0,y1 = r*rh, (r+1)*rh
            yield (r,c), bgr[y0:y1, x0:x1]

def annulus_pixels(cell_bgr, r_in=0.55, r_out=0.88):
    h,w = cell_bgr.shape[:2]
    R = 0.5*min(h,w)
    ri, ro = int(R*r_in), int(R*r_out)
    cy,cx = h//2, w//2
    Y,X = np.ogrid[:h,:w]
    d2 = (X-cx)**2 + (Y-cy)**2
    return cell_bgr[(d2>=ri*ri)&(d2<=ro*ro)]

def classify_cell(cell_bgr):
    pix = annulus_pixels(cell_bgr)
    if pix.size == 0: return ""
    hsv = cv2.cvtColor(pix.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_BGR2HSV)[:,0,:]
    H,S,V = hsv[:,0], hsv[:,1], hsv[:,2]
    nr = IS_RED(H,S,V).sum(); nb = IS_BLUE(H,S,V).sum(); ng = IS_GREEN(H,S,V).sum()
    tot = nr+nb+ng
    if tot == 0: return ""
    if ng/tot >= 0.18: return "T"
    return "P" if nb >= nr else "B"

def parse_bead_from_image(file_bytes):
    img  = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    roi  = find_board_roi(img)
    norm = rectify(roi)
    seq=[]
    for (r,c), patch in split_cells(norm):
        lab = classify_cell(patch)
        if lab: seq.append((c,r,lab))
    out=[]
    for c in range(COLS):
        for cc, rr, lab in [(cc, rr, lab) for cc, rr, lab in seq if cc==c]:
            out.append(lab)
    while out and out[-1] not in ("P","B","T"):
        out.pop()
    return out, norm

def prepare72(history):
    seq = [o for o in history if o in ("P","B","T")]
    seq = seq[-72:]
    return [""]*(72-len(seq)) + seq

def text_bar(p):
    n=int(p*20); return "█"*n + "░"*(20-n)

st.title("Baccarat Predictor")
st.caption("Bead order = top to bottom, then left to right.")

# --- state buckets ---
if "history" not in st.session_state:
    st.session_state.history = []
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "parsed_seq" not in st.session_state:
    st.session_state.parsed_seq = []

st.subheader("Paste from clipboard or upload image")

colA, colB, colC = st.columns([1,1,1])

# 1) Paste
with colA:
    if st.button("Paste from clipboard"):
        grabbed = ImageGrab.grabclipboard()
        if isinstance(grabbed, Image.Image):
            buf = io.BytesIO(); grabbed.save(buf, format="PNG")
            st.session_state.img_bytes = buf.getvalue()
            st.success("Image grabbed from clipboard")
        elif isinstance(grabbed, list) and grabbed:
            try:
                st.session_state.img_bytes = open(grabbed[0], "rb").read()
                st.success(f"Loaded file from clipboard: {grabbed[0]}")
            except Exception as e:
                st.error(f"Clipboard file read failed: {e}")
        else:
            st.warning("Clipboard has no image or file")

# 2) Upload
with colB:
    up = st.file_uploader("…or upload PNG/JPG", type=["png","jpg","jpeg"], key="uploader")
    if up is not None:
        st.session_state.img_bytes = up.read()

# 3) Parse current image
with colC:
    if st.button("Parse image"):
        if st.session_state.img_bytes:
            seq_img, viz = parse_bead_from_image(st.session_state.img_bytes)
            st.session_state.parsed_seq = seq_img
            st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), caption="Detected bead region", use_container_width=True)
            st.code(" ".join(seq_img) or "(no tokens)")
        else:
            st.warning("Paste or upload an image first")

# Show last parsed sequence if available
if st.session_state.parsed_seq:
    st.markdown("**Last parsed tokens:**")
    st.code(" ".join(st.session_state.parsed_seq))
    if st.button("Append parsed to history"):
        st.session_state.history.extend(st.session_state.parsed_seq)
        st.success(f"Appended {len(st.session_state.parsed_seq)} tokens")

# Manual input
st.subheader("Manual input")
user_line = st.text_input("Add outcomes (space separated P/B/T)", "")
c1, c2 = st.columns(2)
if c1.button("Add"):
    toks = [t for t in user_line.upper().split() if t in {"P","B","T"}]
    if toks: st.session_state.history.extend(toks); st.success(f"Added {len(toks)}")
    else: st.warning("No valid tokens.")
if c2.button("Undo last") and st.session_state.history:
    st.session_state.history.pop()
if st.button("Clear all"):
    st.session_state.history.clear()
st.write(f"Current length: {len(st.session_state.history)}")
st.code(" ".join(st.session_state.history[-200:]) or "(empty)")

st.subheader("Model input (prepared 72 tokens)")
seq72 = prepare72(st.session_state["history"])
st.code(" ".join([t if t else "_" for t in seq72]))

if st.button("Predict"):
    try:
        res = pred.predict_with_analysis(st.session_state["history"])
        out = res["prediction"]
        st.subheader("Prediction")
        st.write(f"Next: **{out['predicted_outcome']}**")
        st.write(f"Confidence: **{out['confidence']:.1%}** | Data: {res['game_state']['data_sufficiency']}")
        if "probabilities" in out:
            st.write("Probabilities:")
            for k,v in out["probabilities"].items():
                st.write(f"{k}: {v:.1%}  {text_bar(v)}")
        pat = res["pattern_analysis"]
        st.subheader("Pattern")
        st.write(f"Current streak: {pat['current_streak']['outcome']} × {pat['current_streak']['length']}")
        st.write(f"Last 5: {pat['recent_distribution']['last_5_games']}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
