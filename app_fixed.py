
import io
import json
import re
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from openai import OpenAI

st.set_page_config(layout="wide")
st.title("Domain Categorization App")

PREDEFINED_CATEGORIES = [
    "Retailer",
    "Distributor",
    "Brand",
    "Forum",
    "Marketplace",
    "Comparison Site",
    "Publisher",
    "Organization",
    "Platform",
]

DEFAULT_CATEGORIES = PREDEFINED_CATEGORIES + [
    "Directory",
    "Review Site",
    "Software Vendor",
    "Government",
    "Educational Institution",
    "Community Platform",
]

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
use_openai = st.sidebar.checkbox("Use OpenAI for classification", value=True)
client = OpenAI(api_key=api_key) if api_key else None


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def find_matching_sheet(excel_file: pd.ExcelFile, target_name: str = "citation") -> Optional[str]:
    target = normalize_name(target_name)
    for sheet in excel_file.sheet_names:
        if normalize_name(sheet) == target:
            return sheet
    return None


def find_matching_column(columns, target_name: str = "parent domain") -> Optional[str]:
    target = normalize_name(target_name)
    for col in columns:
        if normalize_name(col) == target:
            return col
    return None


def fetch_domain_text(domain: str):
    urls_to_try = [f"https://{domain}", f"http://{domain}"]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DomainCategorizationBot/1.0)"
    }

    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=8, headers=headers, allow_redirects=True)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            description = ""
            meta = soup.find("meta", attrs={"name": "description"})
            if meta:
                description = meta.get("content", "").strip()

            text = soup.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text)[:2500]
            return title, description, text
        except Exception:
            continue

    return "", "", ""


def rule_based_classification(domain: str, title: str, description: str, text: str):
    blob = " ".join([domain, title, description, text]).lower()

    def has_any(keywords):
        return any(k in blob for k in keywords)

    if has_any(["forum", "thread", "threads", "community discussion", "reply", "replies"]):
        return "Forum", "", "High", "The site appears to center on user discussions and threaded conversations."

    if has_any(["marketplace", "multiple sellers", "sell on", "vendors", "listings", "buy and sell"]):
        return "Marketplace", "", "Medium", "The site appears to host listings or offers from multiple sellers."

    if has_any(["compare prices", "comparison", "best price", "vs.", "versus", "price comparison"]):
        return "Comparison Site", "", "Medium", "The site appears to focus on comparing products, services, or prices."

    if has_any(["news", "review", "reviews", "editorial", "blog", "magazine", "article", "articles"]):
        return "Publisher", "", "Medium", "The site appears to publish editorial, review, or informational content."

    if has_any(["association", "foundation", "ngo", "government", "municipality", "university", ".gov", ".edu"]):
        return "Organization", "", "Medium", "The site appears to belong to a non-commercial institution or public body."

    if has_any(["official site", "official website", "our products", "about us", "manufacturer", "brand"]):
        return "Brand", "", "Medium", "The site appears to represent an official company or manufacturer."

    if has_any(["shop", "cart", "checkout", "buy now", "add to cart", "e-commerce"]):
        return "Retailer", "", "Medium", "The site appears to sell products directly to consumers."

    if has_any(["reseller", "wholesale", "trade account", "dealer", "distributor", "business customers"]):
        return "Distributor", "", "Medium", "The site appears to supply products to businesses or resellers."

    if has_any(["platform", "dashboard", "manage listings", "tool", "software", "service"]):
        return "Platform", "", "Low", "The site appears to offer a digital service or user functionality, but the primary role is not fully clear."

    return "Publisher", "Unclear Website Type", "Low", "The domain could not be classified confidently from lightweight inspection alone."


def classify_with_openai(domain: str, title: str, description: str, text: str):
    if not client or not use_openai:
        return None

    prompt = f"""
You are a domain classification analyst. Classify the domain based on its primary website function.

Predefined categories:
Retailer
Distributor
Brand
Forum
Marketplace
Comparison Site
Publisher
Organization
Platform

Rules:
- Prefer predefined categories whenever possible
- Only suggest a new subcategory when helpful
- Focus on the main purpose of the site, not every feature
- Brand site with webshop should usually be Brand
- Editorial blog with affiliate links should usually be Publisher
- Return valid JSON only
- Allowed confidence values: High, Medium, Low

Domain: {domain}
Title: {title}
Meta description: {description}
Visible text sample: {text}

Return JSON with exactly these keys:
category
subcategory
confidence
reasoning
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)

        return (
            str(data.get("category", "")).strip(),
            str(data.get("subcategory", "")).strip(),
            str(data.get("confidence", "")).strip(),
            str(data.get("reasoning", "")).strip(),
        )
    except Exception:
        return None


def build_download_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="categorized_domains")
    output.seek(0)
    return output.getvalue()


uploaded = st.file_uploader("Upload file", type=["xlsx", "xls", "csv"])

if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded)
        else:
            excel = pd.ExcelFile(uploaded)
            matching_sheet = find_matching_sheet(excel, "citation")

            if not matching_sheet:
                st.error(
                    "No sheet matching 'citation' was found. "
                    f"Available sheets: {', '.join(excel.sheet_names)}"
                )
                st.stop()

            raw_df = pd.read_excel(uploaded, sheet_name=matching_sheet)

        domain_col = find_matching_column(raw_df.columns, "parent domain")
        if not domain_col:
            st.error(
                "No column matching 'parent domain' was found. "
                f"Available columns: {', '.join(map(str, raw_df.columns))}"
            )
            st.stop()

        domains = (
            raw_df[domain_col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        if not domains:
            st.warning("No non-empty parent domains were found in the file.")
            st.stop()

        st.info(f"Found {len(domains)} unique parent domains.")

        progress = st.progress(0)
        results = []

        for idx, domain in enumerate(domains, start=1):
            title, description, text = fetch_domain_text(domain)
            category, subcategory, confidence, reasoning = rule_based_classification(
                domain, title, description, text
            )

            openai_result = classify_with_openai(domain, title, description, text)
            if openai_result:
                category, subcategory, confidence, reasoning = openai_result

            results.append(
                {
                    "Parent Domain": domain,
                    "Category": category,
                    "Suggested Subcategory": subcategory,
                    "Confidence": confidence,
                    "Reasoning": reasoning,
                }
            )
            progress.progress(idx / len(domains))

        result_df = pd.DataFrame(results)

        st.subheader("Results")
        edited_df = st.data_editor(
            result_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Category": st.column_config.SelectboxColumn(
                    "Category",
                    options=DEFAULT_CATEGORIES,
                    required=False,
                ),
                "Confidence": st.column_config.SelectboxColumn(
                    "Confidence",
                    options=["High", "Medium", "Low"],
                    required=False,
                ),
            },
            key="results_editor",
        )

        st.subheader("Summary")
        distribution = (
            edited_df["Category"]
            .fillna("Uncategorized")
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Count")
        )
        st.dataframe(distribution, use_container_width=True)

        new_categories = sorted(
            {
                c for c in edited_df["Category"].dropna().astype(str).str.strip()
                if c and c not in PREDEFINED_CATEGORIES
            }
        )
        st.write("**Newly suggested categories:**", ", ".join(new_categories) if new_categories else "None")

        low_conf_df = edited_df[
            edited_df["Confidence"].fillna("").astype(str).str.strip().str.lower() == "low"
        ][["Parent Domain", "Category", "Reasoning"]]
        st.write("**Low-confidence domains:**")
        if low_conf_df.empty:
            st.write("None")
        else:
            st.dataframe(low_conf_df, use_container_width=True)

        excel_bytes = build_download_excel(edited_df)
        st.download_button(
            "Download Excel",
            data=excel_bytes,
            file_name="categorized_domains.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.exception(e)
