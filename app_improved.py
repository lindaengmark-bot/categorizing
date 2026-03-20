import io
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

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

SUGGESTED_EXTRA_CATEGORIES = [
    "Directory",
    "Review Site",
    "Software Vendor",
    "Government",
    "Educational Institution",
    "Community Platform",
]

ALL_CATEGORIES = PREDEFINED_CATEGORIES + SUGGESTED_EXTRA_CATEGORIES

USER_AGENT = "Mozilla/5.0 (compatible; DomainCategorizationApp/2.0)"
TIMEOUT = 10

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
use_openai = st.sidebar.checkbox("Use OpenAI classification", value=True)
model_name = st.sidebar.text_input("OpenAI model", value="gpt-4.1-mini")
max_domains = st.sidebar.number_input("Max domains to process", min_value=1, max_value=5000, value=250, step=1)
show_debug = st.sidebar.checkbox("Show debug details", value=False)

client = OpenAI(api_key=api_key) if api_key else None


@dataclass
class SiteContext:
    domain: str
    final_url: str
    title: str
    meta_description: str
    headings: List[str]
    nav_items: List[str]
    footer_text: str
    visible_text: str
    sampled_pages: Dict[str, str]
    signals: Dict[str, bool]
    fetch_status: str


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


def normalize_domain(raw: str) -> str:
    value = str(raw).strip().lower()
    value = re.sub(r"^https?://", "", value)
    value = value.split("/")[0]
    return value.strip()


def clean_text(text: str, max_len: int = 3000) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:max_len]


def extract_page_content(soup: BeautifulSoup) -> Tuple[str, str, List[str], List[str], str, str]:
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    meta_description = ""
    meta = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    if meta:
        meta_description = meta.get("content", "").strip()

    headings = [clean_text(h.get_text(" ", strip=True), 200) for h in soup.find_all(["h1", "h2", "h3"])[:12]]
    nav_items = [clean_text(a.get_text(" ", strip=True), 100) for a in soup.select("nav a")[:20]]
    footer = soup.find("footer")
    footer_text = clean_text(footer.get_text(" ", strip=True), 800) if footer else ""

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    visible_text = clean_text(soup.get_text(" ", strip=True), 4000)

    return title, meta_description, headings, nav_items, footer_text, visible_text


def score_signals(text_blob: str, domain: str) -> Dict[str, bool]:
    blob = f"{domain} {text_blob}".lower()

    def has_any(terms: List[str]) -> bool:
        return any(term in blob for term in terms)

    return {
        "has_shop_terms": has_any(["add to cart", "basket", "checkout", "shop now", "buy now", "shopping cart", "order now"]),
        "has_consumer_retail_terms": has_any(["free shipping", "customer reviews", "shop", "our store", "wishlist", "buy online"]),
        "has_brand_terms": has_any(["official site", "official website", "our story", "about us", "our products", "manufacturer", "founded in", "we make", "brand"]),
        "has_distributor_terms": has_any(["wholesale", "trade account", "dealer", "dealers", "reseller", "resellers", "installers", "for professionals", "business customers", "b2b"]),
        "has_marketplace_terms": has_any(["sell on", "sellers", "vendors", "marketplace", "list your product", "browse sellers", "multiple sellers"]),
        "has_forum_terms": has_any(["forum", "forums", "thread", "threads", "post reply", "new topic", "community discussion", "view topic"]),
        "has_comparison_terms": has_any(["compare", "comparison", "versus", "vs.", "best price", "price comparison", "compare products"]),
        "has_publisher_terms": has_any(["news", "blog", "editorial", "magazine", "article", "articles", "latest news", "review", "reviews"]),
        "has_organization_terms": has_any(["association", "foundation", "ngo", "nonprofit", "charity", "government", "ministry", "municipality", "public authority", ".gov", ".edu", "university", "school"]),
        "has_platform_terms": has_any(["software", "saas", "dashboard", "workspace", "manage", "platform", "tool", "tools", "service", "analytics", "login", "sign in"]),
        "has_directory_terms": has_any(["directory", "find providers", "find dealers", "business listing", "browse companies"]),
        "has_review_terms": has_any(["expert review", "user reviews", "rated", "ratings", "top picks"]),
        "has_community_terms": has_any(["community", "members", "groups", "join the community"]),
    }


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_site_context(domain: str) -> SiteContext:
    headers = {"User-Agent": USER_AGENT}
    urls_to_try = [f"https://{domain}", f"http://{domain}"]

    homepage_html = None
    final_url = ""
    status = "failed"

    for url in urls_to_try:
        try:
            resp = requests.get(url, timeout=TIMEOUT, headers=headers, allow_redirects=True)
            resp.raise_for_status()
            homepage_html = resp.text
            final_url = resp.url
            status = "ok"
            break
        except Exception:
            continue

    if homepage_html is None:
        return SiteContext(
            domain=domain,
            final_url="",
            title="",
            meta_description="",
            headings=[],
            nav_items=[],
            footer_text="",
            visible_text="",
            sampled_pages={},
            signals=score_signals("", domain),
            fetch_status=status,
        )

    soup = BeautifulSoup(homepage_html, "html.parser")
    title, meta_description, headings, nav_items, footer_text, visible_text = extract_page_content(soup)

    sampled_pages = {}
    candidate_terms = {
        "about": ["about", "about-us", "our-story", "company"],
        "shop": ["shop", "products", "store", "catalog"],
        "trade": ["dealers", "dealer", "trade", "wholesale", "professional", "resellers", "installers"],
        "forum": ["forum", "community", "threads"],
        "marketplace": ["marketplace", "vendors", "sellers", "sell-on"],
    }

    parsed_home = urlparse(final_url)
    base_netloc = parsed_home.netloc

    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        text = clean_text(a.get_text(" ", strip=True), 80)
        if not href:
            continue
        absolute = urljoin(final_url, href)
        parsed = urlparse(absolute)
        if parsed.netloc != base_netloc:
            continue
        links.append((absolute, f"{text} {absolute}".lower()))

    for label, terms in candidate_terms.items():
        chosen_url = None
        for absolute, combined in links:
            if any(term in combined for term in terms):
                chosen_url = absolute
                break
        if not chosen_url:
            continue
        try:
            resp = requests.get(chosen_url, timeout=TIMEOUT, headers=headers, allow_redirects=True)
            resp.raise_for_status()
            page_soup = BeautifulSoup(resp.text, "html.parser")
            _, _, extra_headings, extra_nav_items, extra_footer, extra_text = extract_page_content(page_soup)
            sampled_pages[label] = clean_text(" ".join(extra_headings + extra_nav_items + [extra_footer, extra_text]), 1500)
        except Exception:
            continue

    combined_text = " ".join(
        [title, meta_description, " ".join(headings), " ".join(nav_items), footer_text, visible_text]
        + list(sampled_pages.values())
    )
    signals = score_signals(combined_text, domain)

    return SiteContext(
        domain=domain,
        final_url=final_url,
        title=title,
        meta_description=meta_description,
        headings=headings,
        nav_items=nav_items,
        footer_text=footer_text,
        visible_text=visible_text,
        sampled_pages=sampled_pages,
        signals=signals,
        fetch_status=status,
    )


def rule_based_classification(ctx: SiteContext):
    scores = {category: 0 for category in PREDEFINED_CATEGORIES}
    s = ctx.signals

    if s["has_forum_terms"]:
        scores["Forum"] += 9
    if s["has_marketplace_terms"]:
        scores["Marketplace"] += 8
    if s["has_comparison_terms"]:
        scores["Comparison Site"] += 8
    if s["has_distributor_terms"]:
        scores["Distributor"] += 8
    if s["has_shop_terms"] or s["has_consumer_retail_terms"]:
        scores["Retailer"] += 7
    if s["has_brand_terms"]:
        scores["Brand"] += 8
    if s["has_publisher_terms"]:
        scores["Publisher"] += 5
    if s["has_organization_terms"]:
        scores["Organization"] += 8
    if s["has_platform_terms"]:
        scores["Platform"] += 6

    if s["has_brand_terms"] and (s["has_shop_terms"] or s["has_consumer_retail_terms"]):
        scores["Brand"] += 4
    if s["has_marketplace_terms"] and s["has_platform_terms"]:
        scores["Marketplace"] += 2
    if s["has_distributor_terms"] and s["has_shop_terms"]:
        scores["Distributor"] += 1
    if s["has_publisher_terms"] and any([s["has_brand_terms"], s["has_shop_terms"], s["has_distributor_terms"], s["has_marketplace_terms"]]):
        scores["Publisher"] -= 2
    if s["has_platform_terms"] and any([s["has_brand_terms"], s["has_marketplace_terms"], s["has_distributor_terms"]]):
        scores["Platform"] -= 2

    best_category = max(scores, key=scores.get)
    top_score = scores[best_category]
    sorted_scores = sorted(scores.values(), reverse=True)
    gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

    if ctx.fetch_status != "ok":
        return "Publisher", "Unclear Website Type", "Low", "The website could not be fetched reliably, so the result is a weak fallback.", scores

    if top_score >= 10 and gap >= 3:
        confidence = "High"
    elif top_score >= 7:
        confidence = "Medium"
    else:
        confidence = "Low"

    reasoning_map = {
        "Retailer": "The site shows consumer shopping signals and appears to sell directly to end customers.",
        "Distributor": "The site shows trade, dealer, installer, wholesale, or reseller signals, which suggests a B2B supply role.",
        "Brand": "The site appears to be the official presence of one company or manufacturer, even if it also has ecommerce features.",
        "Forum": "The site appears centered on discussion threads, replies, and community conversations.",
        "Marketplace": "The site appears to host listings or offers from multiple sellers rather than representing a single company.",
        "Comparison Site": "The site appears focused on comparing products, services, or prices.",
        "Publisher": "The site appears primarily editorial or informational, such as news, blog, or reviews.",
        "Organization": "The site appears to belong to a non-commercial institution, association, government body, or educational entity.",
        "Platform": "The site appears to provide a digital service, tool, or interactive environment.",
    }

    subcategory = ""
    if s["has_directory_terms"] and best_category == "Platform":
        subcategory = "Directory"
    elif s["has_review_terms"] and best_category == "Publisher":
        subcategory = "Review Site"
    elif ".gov" in ctx.domain and best_category == "Organization":
        subcategory = "Government"
    elif ".edu" in ctx.domain and best_category == "Organization":
        subcategory = "Educational Institution"

    return best_category, subcategory, confidence, reasoning_map[best_category], scores


def build_openai_prompt(ctx: SiteContext, rule_result):
    rule_category, rule_subcategory, rule_confidence, rule_reasoning, rule_scores = rule_result

    return f"""
You are a domain classification analyst. Your task is to classify one domain by its PRIMARY website function.

Use these categories whenever possible:
Retailer
Distributor
Brand
Forum
Marketplace
Comparison Site
Publisher
Organization
Platform

Definitions:
Retailer: E-commerce website selling products directly to consumers.
Distributor: Website supplying products to resellers, installers, or businesses.
Brand: Official website representing a manufacturer or company.
Forum: User discussion platform with threads, questions, and replies.
Marketplace: Platform where multiple sellers list products or services.
Comparison Site: Website comparing products, services, or prices.
Publisher: Editorial or informational content site, such as news, blog, or review site.
Organization: Non-commercial entity such as NGO, association, educational body, or government.
Platform: Digital service enabling tools, listings, or user interaction.

Important instructions:
- Prefer the predefined categories.
- Focus on the main purpose of the website, not every feature.
- Do not overuse Publisher. A site is not Publisher just because it has articles or blog posts.
- Do not overuse Platform. A site is not Platform just because it has login, tools, or software elements.
- If the site primarily represents one company or manufacturer, prefer Brand.
- A brand site with a webshop should usually still be Brand.
- If the site mainly serves dealers, installers, trade buyers, or resellers, prefer Distributor.
- If the site hosts many sellers or vendors, prefer Marketplace.
- Only suggest a new subcategory when useful.
- Allowed confidence values: High, Medium, Low.
- Return valid JSON only.

Useful examples:
- Official manufacturer website with ecommerce -> Brand
- Editorial review site with affiliate links -> Publisher, subcategory Review Site
- Wholesale site for installers -> Distributor
- User thread site -> Forum
- Site comparing products and prices -> Comparison Site
- Association or public institution -> Organization
- Multi-seller commerce site -> Marketplace

Domain: {ctx.domain}
Resolved URL: {ctx.final_url}
Homepage title: {ctx.title}
Meta description: {ctx.meta_description}
Headings: {json.dumps(ctx.headings[:10], ensure_ascii=False)}
Navigation labels: {json.dumps(ctx.nav_items[:15], ensure_ascii=False)}
Footer text: {ctx.footer_text}
Homepage text snippet: {ctx.visible_text[:2200]}

Sampled internal pages:
{json.dumps(ctx.sampled_pages, ensure_ascii=False)}

Extracted signals:
{json.dumps(ctx.signals, ensure_ascii=False)}

Rule-based starting point:
category={rule_category}
subcategory={rule_subcategory}
confidence={rule_confidence}
reasoning={rule_reasoning}
scores={rule_scores}

Return JSON with exactly these keys:
category
subcategory
confidence
reasoning
"""


def classify_with_openai(ctx: SiteContext, rule_result):
    if not client or not use_openai:
        return None

    prompt = build_openai_prompt(ctx, rule_result)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a careful analyst. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL).strip()

        data = json.loads(content)
        category = str(data.get("category", "")).strip()
        subcategory = str(data.get("subcategory", "")).strip()
        confidence = str(data.get("confidence", "")).strip()
        reasoning = str(data.get("reasoning", "")).strip()

        if category not in ALL_CATEGORIES:
            category = rule_result[0]
        if confidence not in ["High", "Medium", "Low"]:
            confidence = rule_result[2]
        if not reasoning:
            reasoning = rule_result[3]

        return category, subcategory, confidence, reasoning
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
                st.error("No sheet matching 'citation' was found. Available sheets: " + ", ".join(excel.sheet_names))
                st.stop()
            raw_df = pd.read_excel(uploaded, sheet_name=matching_sheet)

        domain_col = find_matching_column(raw_df.columns, "parent domain")
        if not domain_col:
            st.error("No column matching 'parent domain' was found. Available columns: " + ", ".join(map(str, raw_df.columns)))
            st.stop()

        domains = (
            raw_df[domain_col]
            .dropna()
            .astype(str)
            .map(normalize_domain)
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        if not domains:
            st.warning("No non-empty parent domains were found in the file.")
            st.stop()

        if len(domains) > max_domains:
            st.warning(f"Only the first {max_domains} domains will be processed in this run.")
            domains = domains[:max_domains]

        st.info(f"Found {len(domains)} unique parent domains.")

        results = []
        progress = st.progress(0.0)

        for idx, domain in enumerate(domains, start=1):
            ctx = fetch_site_context(domain)
            rule_result = rule_based_classification(ctx)
            final_result = classify_with_openai(ctx, rule_result) or rule_result[:4]

            results.append({
                "Parent Domain": domain,
                "Category": final_result[0],
                "Suggested Subcategory": final_result[1],
                "Confidence": final_result[2],
                "Reasoning": final_result[3],
                "Fetched URL": ctx.final_url,
                "Fetch Status": ctx.fetch_status,
            })
            progress.progress(idx / len(domains))

        result_df = pd.DataFrame(results)

        st.subheader("Results")
        edited_df = st.data_editor(
            result_df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Category": st.column_config.SelectboxColumn("Category", options=ALL_CATEGORIES, required=False),
                "Confidence": st.column_config.SelectboxColumn("Confidence", options=["High", "Medium", "Low"], required=False),
                "Fetched URL": st.column_config.TextColumn("Fetched URL", disabled=True),
                "Fetch Status": st.column_config.TextColumn("Fetch Status", disabled=True),
            },
            key="results_editor",
        )

        st.subheader("Summary")
        distribution = (
            edited_df["Category"]
            .fillna("Uncategorized")
            .astype(str)
            .str.strip()
            .replace("", "Uncategorized")
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Count")
        )
        st.dataframe(distribution, use_container_width=True, hide_index=True)

        new_categories = sorted({
            value for value in edited_df["Category"].fillna("").astype(str).str.strip()
            if value and value not in PREDEFINED_CATEGORIES
        })
        st.write("**Newly suggested categories:**", ", ".join(new_categories) if new_categories else "None")

        low_conf_df = edited_df[
            edited_df["Confidence"].fillna("").astype(str).str.strip().str.lower() == "low"
        ][["Parent Domain", "Category", "Reasoning"]]
        st.write("**Low-confidence domains:**")
        if low_conf_df.empty:
            st.write("None")
        else:
            st.dataframe(low_conf_df, use_container_width=True, hide_index=True)

        if show_debug:
            st.subheader("Debug guidance")
            st.write(
                "If too many rows still land in Publisher or Platform, check whether the fetched URL is correct, "
                "whether the site blocks requests, and whether the model had enough useful context."
            )

        excel_bytes = build_download_excel(edited_df)
        st.download_button(
            "Download Excel",
            data=excel_bytes,
            file_name="categorized_domains.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.exception(e)
