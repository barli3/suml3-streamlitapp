import streamlit as st
import pandas as pd
import os
import time

st.set_page_config(
    page_title="SUML3 - s29371",
    layout="centered"
)

st.image("https://upload.wikimedia.org/wikipedia/commons/d/d6/Hf-logo-with-title.svg")
st.title("SUML3 - s29371")
st.markdown("---")

st.info(
    "**Do czego służy ta aplikacja?**\n\n"
    "Aplikacja wykorzystuje modele językowe z biblioteki Hugging Face do dwóch zadań:\n"
    "- **Analiza wydźwięku** – sprawdź, czy tekst (ang.) jest pozytywny czy negatywny\n"
    "- **Tłumaczenie EN > DE** – przetłumacz tekst z angielskiego na niemiecki\n\n"
    "Wybierz opcję z listy poniżej i wpisz tekst w odpowiednim polu."
)

st.markdown("---")

st.header("Podgląd DSP_4.csv")

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DSP_4.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    st.dataframe(df)
    st.caption(f"Wczytano {len(df)} rekordów z pliku DSP_4.csv")
else:
    st.warning("Nie znaleziono pliku DSP_4.csv, upewnij się, że jest w tym samym katalogu co skrypt.")

st.markdown("---")

st.header("Przetwarzanie języka naturalnego")
st.write("Wybierz jedno z dostępnych opcji:")

if "poprzednia_opcja" not in st.session_state:
    st.session_state.poprzednia_opcja = None
if "model_zaladowany" not in st.session_state:
    st.session_state.model_zaladowany = False

option = st.selectbox(
    "Wybierz opcję:",
    [
        "Wydźwięk emocjonalny tekstu (ang.)",
        "Tłumaczenie tekstu: angielski > niemiecki",
    ],
)

if option != st.session_state.poprzednia_opcja:
    st.session_state.poprzednia_opcja = option
    st.session_state.model_zaladowany = False

if not st.session_state.model_zaladowany:
    with st.spinner("Ładowanie modelu..."):
        time.sleep(1.5)
    st.session_state.model_zaladowany = True
    st.rerun()

if option == "Wydźwięk emocjonalny tekstu (ang.)":
    st.subheader("Analiza wydźwięku emocjonalnego")
    st.write("Wpisz dowolny tekst w języku **angielskim**, a model oceni czy jest pozytywny czy negatywny.")
    st.success("Model gotowy! Wpisz tekst poniżej:")

    text = st.text_area("Wpisz tekst tutaj:", placeholder="np. Hello world!")

    if text:
        with st.spinner("Analizuję tekst..."):
            try:
                from transformers import pipeline
                classifier = pipeline("sentiment-analysis")
                answer = classifier(text)
                label = answer[0]["label"]
                score = round(answer[0]["score"] * 100, 2)

                if label == "POSITIVE":
                    st.success(f"Wydźwięk: **POZYTYWNY** (pewność: {score}%)")
                else:
                    st.error(f"Wydźwięk: **NEGATYWNY** (pewność: {score}%)")

                st.json(answer)
            except Exception as e:
                st.error(f"Coś poszło nie tak: {e}")

elif option == "Tłumaczenie tekstu: angielski > niemiecki":
    st.subheader("Tłumaczenie: angielski > niemiecki")
    st.write("Wpisz tekst w języku **angielskim**, a model przetłumaczy go na **niemiecki**.")
    st.success("Model gotowy! Wpisz tekst poniżej:")

    text_to_translate = st.text_area("Wpisz tekst do przetłumaczenia:", placeholder="np. Good morning!")

    if text_to_translate:
        with st.spinner("Tłumaczę..."):
            try:
                from transformers import MarianMTModel, MarianTokenizer
                model_name = "Helsinki-NLP/opus-mt-en-de"
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                tokens = tokenizer([text_to_translate], return_tensors="pt", padding=True)
                translated_tokens = model.generate(**tokens)
                translated = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

                st.success("Tłumaczenie zakończone sukcesem!")
                st.markdown("**Wynik tłumaczenia:**")
                st.info(translated)
            except Exception as e:
                st.error(f"Błąd podczas tłumaczenia: {e}")

st.markdown("---")
st.caption("Numer indeksu: s29371")