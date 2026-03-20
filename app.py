import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

st.set_page_config(layout="wide")
st.title("Domain Categorization App")

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
client = OpenAI(api_key=api_key) if api_key else None

CATEGORIES = [
    "Retailer","Distributor","Brand","Forum","Marketplace",
    "Comparison Site","Publisher","Organization","Platform"
]

def fetch_domain_text(domain):
    try:
        url = f"https://{domain}"
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string if soup.title else ""
        desc = ""
        meta = soup.find("meta", attrs={"name":"description"})
        if meta:
            desc = meta.get("content","")
        text = soup.get_text(" ", strip=True)[:1000]
        return title, desc, text
    except:
        return "", "", ""

def classify_with_openai(domain, title, desc, text):
    if not client:
        return None

    prompt = f"""Classify this domain: {domain}
Title: {title}
Description: {desc}
Text: {text}

Categories: {CATEGORIES}

Return JSON with keys:
category, subcategory, confidence, reasoning
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        content = response.choices[0].message.content
        return eval(content)
    except:
        return None

uploaded = st.file_uploader("Upload file")

if uploaded:
    df = pd.read_excel(uploaded, sheet_name="citation")
    domains = df["parent domain"].dropna().unique()

    results = []

    for d in domains:
        title, desc, text = fetch_domain_text(d)
        res = classify_with_openai(d, title, desc, text)

        if res:
            results.append({
                "Parent Domain": d,
                "Category": res.get("category",""),
                "Suggested Subcategory": res.get("subcategory",""),
                "Confidence": res.get("confidence",""),
                "Reasoning": res.get("reasoning","")
            })
        else:
            results.append({
                "Parent Domain": d,
                "Category": "",
                "Suggested Subcategory": "",
                "Confidence": "Low",
                "Reasoning": "Failed classification"
            })

    result_df = pd.DataFrame(results)

    st.data_editor(result_df, use_container_width=True)

    st.download_button(
        "Download CSV",
        result_df.to_csv(index=False),
        "results.csv"
    )
