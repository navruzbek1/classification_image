import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
st.title("RASMLARNI TANIYDIGAN DASTUR(ruhsat etilgan imglar uchun 'Telephone','Biycle','Clock')")

#rasmni joylash
file = st.file_uploader("Rasm yuklash",type=['png','jpeg','gif','svg'])

if file:
    st.image(file)
    img = PILImage.create(file)


    model = load_learner('exper_model.pkl')

    pred, pred_id, probs = model.predict(img)
    st.success(pred)
    st.info(f'Aniqligi: {probs[pred_id]*100:.1f}%')

    #plotling
    fig =  px.bar(x=probs*100,y=model.dls.vocab)
    st.plotly_chart(fig)