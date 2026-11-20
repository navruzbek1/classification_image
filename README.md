# classification_image

Rasmni tanib beruvchi ilova: **fastai** modeli yordamida **telefon, velosiped va soat** rasmlarini klassifikatsiya qiladi.

## Tarkib

- `app.py`: Streamlit ilovasi. Rasm yuklanadi, model sinfni, ishonchlilikni (%) va barcha sinflar ehtimoli diagrammasini (Plotly) ko'rsatadi.
- `exper_model.pkl`: o'qitilgan fastai modeli (~87 MB).

## Ishga tushirish

```bash
pip install streamlit fastai plotly
streamlit run app.py
```

## Eslatma

- Model ilovada ko'rsatilgan sinflar uchun mo'ljallangan (telefon, velosiped, soat).
- Model Windows'da saqlangani uchun `app.py` Linux'da `pathlib` yo'llarini moslashtiradi.
