import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

st.set_page_config(page_title="SmartFeed Pro", layout="wide")

PROFILE_DEFAULTS = {
    "Layer Peak": {"Total inclusion": 100.0, "CP %": 18.0, "ME kcal/kg": 2800.0, "Lys %": 0.90, "Met %": 0.42, "Ca %": 4.00, "AvP %": 0.45, "Max Fiber %": 7.0, "Target Cost/kg": 0.42},
    "Layer Late": {"Total inclusion": 100.0, "CP %": 16.5, "ME kcal/kg": 2750.0, "Lys %": 0.78, "Met %": 0.36, "Ca %": 4.10, "AvP %": 0.40, "Max Fiber %": 7.5, "Target Cost/kg": 0.39},
    "Broiler Starter": {"Total inclusion": 100.0, "CP %": 22.0, "ME kcal/kg": 3000.0, "Lys %": 1.20, "Met %": 0.52, "Ca %": 1.00, "AvP %": 0.50, "Max Fiber %": 5.5, "Target Cost/kg": 0.48},
    "Broiler Grower": {"Total inclusion": 100.0, "CP %": 20.0, "ME kcal/kg": 3150.0, "Lys %": 1.05, "Met %": 0.45, "Ca %": 0.90, "AvP %": 0.45, "Max Fiber %": 6.0, "Target Cost/kg": 0.46},
    "Custom": {"Total inclusion": 100.0, "CP %": 18.0, "ME kcal/kg": 2800.0, "Lys %": 0.90, "Met %": 0.42, "Ca %": 4.00, "AvP %": 0.45, "Max Fiber %": 7.0, "Target Cost/kg": 0.42},
}
WEIGHTS_DEFAULT = {"Nutrition adequacy": 0.40, "Gut score": 0.25, "Cost score": 0.20, "Energy:protein balance": 0.15}

MARKET_DEFAULTS = pd.DataFrame([
    {"Commodity": "Corn Futures", "Raw Market Quote": 4.4874, "Quote Unit": "USD/bushel corn", "Last Checked": "2026-04-16", "Source URL": "https://tradingeconomics.com/commodity/corn", "Notes": "Editable benchmark."},
    {"Commodity": "Soybean Meal Futures", "Raw Market Quote": 331.6, "Quote Unit": "USD/metric ton", "Last Checked": "2026-04-16", "Source URL": "https://markets.businessinsider.com/commodities/soybean-meal-price", "Notes": "Editable benchmark."},
    {"Commodity": "Soybean Oil", "Raw Market Quote": 0.69, "Quote Unit": "USD/lb", "Last Checked": "2026-04-16", "Source URL": "https://markets.businessinsider.com/commodities/soybean-oil-price", "Notes": "Editable benchmark."},
    {"Commodity": "Fish Meal", "Raw Market Quote": 1.45, "Quote Unit": "USD/kg", "Last Checked": "2026-04-16", "Source URL": "", "Notes": "Placeholder benchmark."},
])

INGREDIENT_DEFAULTS = pd.DataFrame([
    {"Ingredient": "Corn", "Category": "Energy", "Inclusion %": 58.0, "Price Mode": "Benchmark", "Manual Price/kg": 0.32, "Benchmark Commodity": "Corn Futures", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 45.0, "Max %": 65.0, "CP %": 8.5, "ME kcal/kg": 3350, "Lys %": 0.26, "Met %": 0.18, "Ca %": 0.02, "AvP %": 0.08, "Fiber %": 2.2, "Gut Score 0-10": 5.0, "Digestibility 0-10": 7.0, "Notes": "Energy base ingredient", "Source URL": "https://tradingeconomics.com/commodity/corn"},
    {"Ingredient": "Soybean Meal", "Category": "Protein", "Inclusion %": 24.0, "Price Mode": "Benchmark", "Manual Price/kg": 0.52, "Benchmark Commodity": "Soybean Meal Futures", "Conversion Factor": 1.0, "Premium Adj/kg": 0.05, "Min %": 15.0, "Max %": 35.0, "CP %": 46.0, "ME kcal/kg": 2450, "Lys %": 2.90, "Met %": 0.62, "Ca %": 0.30, "AvP %": 0.25, "Fiber %": 3.5, "Gut Score 0-10": 6.0, "Digestibility 0-10": 8.0, "Notes": "Main protein source", "Source URL": "https://markets.businessinsider.com/commodities/soybean-meal-price"},
    {"Ingredient": "Wheat Bran", "Category": "Fiber/Byproduct", "Inclusion %": 5.0, "Price Mode": "Manual", "Manual Price/kg": 0.24, "Benchmark Commodity": "", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 0.0, "Max %": 12.0, "CP %": 15.5, "ME kcal/kg": 1700, "Lys %": 0.55, "Met %": 0.25, "Ca %": 0.13, "AvP %": 0.95, "Fiber %": 10.0, "Gut Score 0-10": 6.0, "Digestibility 0-10": 5.0, "Notes": "Use cautiously because of fiber", "Source URL": ""},
    {"Ingredient": "Soybean Oil", "Category": "Fat", "Inclusion %": 3.0, "Price Mode": "Benchmark", "Manual Price/kg": 1.45, "Benchmark Commodity": "Soybean Oil", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 0.0, "Max %": 8.0, "CP %": 0.0, "ME kcal/kg": 8800, "Lys %": 0.00, "Met %": 0.00, "Ca %": 0.00, "AvP %": 0.00, "Fiber %": 0.0, "Gut Score 0-10": 4.0, "Digestibility 0-10": 9.0, "Notes": "Energy dense fat source", "Source URL": ""},
    {"Ingredient": "Limestone", "Category": "Mineral", "Inclusion %": 8.5, "Price Mode": "Manual", "Manual Price/kg": 0.08, "Benchmark Commodity": "", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 6.0, "Max %": 10.0, "CP %": 0.0, "ME kcal/kg": 0, "Lys %": 0.00, "Met %": 0.00, "Ca %": 38.00, "AvP %": 0.00, "Fiber %": 0.0, "Gut Score 0-10": 5.0, "Digestibility 0-10": 7.0, "Notes": "Calcium source", "Source URL": ""},
    {"Ingredient": "DCP", "Category": "Mineral", "Inclusion %": 1.0, "Price Mode": "Manual", "Manual Price/kg": 0.75, "Benchmark Commodity": "", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 0.0, "Max %": 3.0, "CP %": 0.0, "ME kcal/kg": 0, "Lys %": 0.00, "Met %": 0.00, "Ca %": 23.00, "AvP %": 18.00, "Fiber %": 0.0, "Gut Score 0-10": 5.0, "Digestibility 0-10": 7.0, "Notes": "Available phosphorus source", "Source URL": ""},
    {"Ingredient": "Salt", "Category": "Mineral", "Inclusion %": 0.3, "Price Mode": "Manual", "Manual Price/kg": 0.09, "Benchmark Commodity": "", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 0.2, "Max %": 0.5, "CP %": 0.0, "ME kcal/kg": 0, "Lys %": 0.00, "Met %": 0.00, "Ca %": 0.00, "AvP %": 0.00, "Fiber %": 0.0, "Gut Score 0-10": 5.0, "Digestibility 0-10": 7.0, "Notes": "Sodium source", "Source URL": ""},
    {"Ingredient": "Vitamin-Min Premix", "Category": "Additive", "Inclusion %": 0.2, "Price Mode": "Manual", "Manual Price/kg": 2.10, "Benchmark Commodity": "", "Conversion Factor": 1.0, "Premium Adj/kg": 0.00, "Min %": 0.1, "Max %": 0.5, "CP %": 0.0, "ME kcal/kg": 0, "Lys %": 0.00, "Met %": 0.00, "Ca %": 0.00, "AvP %": 0.00, "Fiber %": 0.0, "Gut Score 0-10": 5.0, "Digestibility 0-10": 7.0, "Notes": "Keep fixed within supplier recommendation", "Source URL": ""},
])

NUMERIC_COLS = ["Inclusion %","Manual Price/kg","Conversion Factor","Premium Adj/kg","Min %","Max %","CP %","ME kcal/kg","Lys %","Met %","Ca %","AvP %","Fiber %","Gut Score 0-10","Digestibility 0-10"]
MIN_NUTRIENTS = ["CP %", "ME kcal/kg", "Lys %", "Met %", "Ca %", "AvP %"]

def render_overview(title_text: str):
    st.markdown(f"""
    <h1 style='font-size:46px; font-weight:800; margin-bottom:0;'>{title_text}</h1>
    <p style='font-size:18px; color:#6b7280; margin-top:6px;'>
    AI-powered poultry feed formulation, diagnostic intelligence, and optimization platform
    </p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(135deg,#0f172a 0%, #1e293b 50%, #334155 100%); border-radius:20px; padding:30px; color:white; margin:10px 0 18px 0;">
        <div style="font-size:30px; font-weight:800; margin-bottom:10px;">From feed calculator to decision engine</div>
        <div style="font-size:16px; max-width:950px; line-height:1.65;">
        Build formulas, pressure-test feasibility, understand bottlenecks, compare cost structure, and optimize with one merged workflow.
        This version is designed to be easier to understand, easier to debug, and easier to use with real ingredient systems.
        </div>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""### 💰 Cost Intelligence
- Manual + benchmark-linked pricing  
- Cost contribution by ingredient  
- Least-cost formulation engine""")
    with c2:
        st.markdown("""### 🧠 Diagnostic Intelligence
- Detect blocked nutrients  
- Explain infeasibility clearly  
- Show what to change first""")
    with c3:
        st.markdown("""### 📊 Research & Decision Support
- Gut score and FEI  
- Nutrient adequacy check  
- Ingredient mix dashboard""")
    st.markdown("---")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.info("**Step 1**  \nSet up ingredients, nutrient values, bounds, and prices in **Ingredient DB**.")
    with s2:
        st.info("**Step 2**  \nOpen **Diagnostic** first to understand whether your formula is mathematically possible.")
    with s3:
        st.info("**Step 3**  \nAfter fixing blockers, run **Optimizer** to get the least-cost feasible formulation.")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Workflow", "Diagnose → Optimize")
    a2.metric("Ingredient control", "Add / Delete")
    a3.metric("Pricing", "Manual + Benchmark")
    a4.metric("Output", "Dashboard + Export")
    st.markdown("---")
    st.warning("This is a decision-support tool. A formula can be mathematically feasible and still need biological validation, supplier confirmation, and practical field review.")

def convert_to_usd_per_kg(raw_quote, unit):
    if pd.isna(raw_quote): return np.nan
    if unit == "USD/bushel corn": return raw_quote / 25.401
    if unit == "USD/bushel soybeans": return raw_quote / 27.216
    if unit == "USD/metric ton": return raw_quote / 1000.0
    if unit == "USD/lb": return raw_quote * 2.20462
    return raw_quote

def prepare_market_df(df):
    out = df.copy()
    out["Benchmark USD/kg"] = out.apply(lambda r: convert_to_usd_per_kg(r["Raw Market Quote"], r["Quote Unit"]), axis=1)
    return out

def prepare_ingredient_df(ingredients, market):
    bench = dict(zip(market["Commodity"], market["Benchmark USD/kg"]))
    out = ingredients.copy()
    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["Ingredient"] = out["Ingredient"].fillna("").astype(str)
    out["Price Mode"] = out["Price Mode"].fillna("Manual")
    out["Benchmark Commodity"] = out["Benchmark Commodity"].fillna("")
    out["Active Price/kg"] = np.where(
        out["Price Mode"].eq("Benchmark"),
        out["Benchmark Commodity"].map(bench).fillna(out["Manual Price/kg"]) * out["Conversion Factor"] + out["Premium Adj/kg"],
        out["Manual Price/kg"]
    )
    return out

def current_targets(selected_profile, custom_targets):
    return custom_targets if selected_profile == "Custom" else PROFILE_DEFAULTS[selected_profile]

def weighted_average(df, col):
    return (df["Inclusion %"] * df[col]).sum() / 100.0

def compute_metrics(ingredients, targets, weights):
    df = ingredients.copy()
    total = df["Inclusion %"].sum()
    if total <= 0: total = 1e-9
    cost = weighted_average(df, "Active Price/kg")
    cp = weighted_average(df, "CP %")
    me = weighted_average(df, "ME kcal/kg")
    lys = weighted_average(df, "Lys %")
    met = weighted_average(df, "Met %")
    ca = weighted_average(df, "Ca %")
    avp = weighted_average(df, "AvP %")
    fiber = weighted_average(df, "Fiber %")
    gut = max(0.0, min(10.0, (0.65 * weighted_average(df, "Gut Score 0-10")) + (0.35 * weighted_average(df, "Digestibility 0-10")) - max(0, fiber - targets["Max Fiber %"]) * 0.6))
    adequacy_scores = [
        min(total / targets["Total inclusion"], 1.0) * 100 if targets["Total inclusion"] else 100,
        min(cp / targets["CP %"], 1.0) * 100 if targets["CP %"] else 100,
        min(me / targets["ME kcal/kg"], 1.0) * 100 if targets["ME kcal/kg"] else 100,
        min(lys / targets["Lys %"], 1.0) * 100 if targets["Lys %"] else 100,
        min(met / targets["Met %"], 1.0) * 100 if targets["Met %"] else 100,
        min(ca / targets["Ca %"], 1.0) * 100 if targets["Ca %"] else 100,
        min(avp / targets["AvP %"], 1.0) * 100 if targets["AvP %"] else 100,
        min(targets["Max Fiber %"] / fiber, 1.0) * 100 if fiber > 0 else 100,
    ]
    nutrition_adequacy = round(float(np.mean(adequacy_scores)), 1)
    target_ep = targets["ME kcal/kg"] / targets["CP %"] if targets["CP %"] else 0
    actual_ep = me / cp if cp else 0
    ep_balance = 0 if target_ep == 0 else max(0.0, 100.0 - abs(actual_ep - target_ep) / target_ep * 100.0)
    cost_score = 0 if cost == 0 else min(targets["Target Cost/kg"] / cost, 1.0) * 100.0
    feed_eff = (weights["Nutrition adequacy"] * nutrition_adequacy + weights["Gut score"] * (gut * 10.0) + weights["Cost score"] * cost_score + weights["Energy:protein balance"] * ep_balance) / sum(weights.values())
    nutrient_table = pd.DataFrame([
        {"Metric":"Total inclusion","Achieved":total,"Target":targets["Total inclusion"],"Status":"OK" if abs(total-targets["Total inclusion"])<=0.01 else "CHECK"},
        {"Metric":"Crude Protein","Achieved":cp,"Target":targets["CP %"],"Status":"OK" if cp>=targets["CP %"] else "CHECK"},
        {"Metric":"ME","Achieved":me,"Target":targets["ME kcal/kg"],"Status":"OK" if me>=targets["ME kcal/kg"] else "CHECK"},
        {"Metric":"Lysine","Achieved":lys,"Target":targets["Lys %"],"Status":"OK" if lys>=targets["Lys %"] else "CHECK"},
        {"Metric":"Methionine","Achieved":met,"Target":targets["Met %"],"Status":"OK" if met>=targets["Met %"] else "CHECK"},
        {"Metric":"Calcium","Achieved":ca,"Target":targets["Ca %"],"Status":"OK" if ca>=targets["Ca %"] else "CHECK"},
        {"Metric":"Available P","Achieved":avp,"Target":targets["AvP %"],"Status":"OK" if avp>=targets["AvP %"] else "CHECK"},
        {"Metric":"Fiber (max)","Achieved":fiber,"Target":targets["Max Fiber %"],"Status":"OK" if fiber<=targets["Max Fiber %"] else "CHECK"},
    ])
    return {"cost":cost,"total":total,"cp":cp,"me":me,"lys":lys,"met":met,"ca":ca,"avp":avp,"fiber":fiber,"gut":gut,"nutrition_adequacy":nutrition_adequacy,"ep_balance":ep_balance,"cost_score":cost_score,"feed_eff":feed_eff,"nutrient_table":nutrient_table}

def nutrient_capacity_analysis(df, targets):
    rows = []
    for nutrient in MIN_NUTRIENTS:
        max_possible = (df["Max %"] * df[nutrient]).sum() / 100.0
        min_locked = (df["Min %"] * df[nutrient]).sum() / 100.0
        gap = max_possible - targets[nutrient]
        status = "OK" if max_possible >= targets[nutrient] else "BLOCKED"
        rows.append({"Nutrient":nutrient,"Target":targets[nutrient],"Max achievable under current bounds":round(max_possible,4),"Locked minimum contribution":round(min_locked,4),"Gap to target":round(gap,4),"Status":status})
    fiber_min_possible = (df["Min %"] * df["Fiber %"]).sum() / 100.0
    rows.append({"Nutrient":"Fiber % (max)","Target":targets["Max Fiber %"],"Max achievable under current bounds":np.nan,"Locked minimum contribution":round(fiber_min_possible,4),"Gap to target":round(targets["Max Fiber %"] - fiber_min_possible,4),"Status":"OK" if fiber_min_possible <= targets["Max Fiber %"] else "BLOCKED"})
    return pd.DataFrame(rows)

def top_contributors(df, column, top_n=3):
    out = df[["Ingredient","Max %",column]].copy()
    out["Potential"] = out["Max %"] * out[column] / 100.0
    return out.sort_values("Potential", ascending=False).head(top_n)[["Ingredient","Potential"]]

def diagnose_feasibility(df, targets):
    issues, fixes, summary = [], [], []
    min_total = df["Min %"].sum()
    max_total = df["Max %"].sum()
    if min_total > targets["Total inclusion"]:
        issues.append(f"Minimum inclusions already total {min_total:.2f}%, above the required {targets['Total inclusion']:.2f}%.")
        fixes.append("Reduce one or more Min % values.")
    if max_total < targets["Total inclusion"]:
        issues.append(f"Maximum inclusions total only {max_total:.2f}%, below the required {targets['Total inclusion']:.2f}%.")
        fixes.append("Increase one or more Max % values so the formula can reach 100%.")
    cap = nutrient_capacity_analysis(df, targets)
    blocked = cap[cap["Status"] == "BLOCKED"]
    for _, r in blocked.iterrows():
        if r["Nutrient"] == "Fiber % (max)":
            issues.append(f"Fiber is blocked: locked minimum fiber contribution is {r['Locked minimum contribution']:.2f}, above the max target of {r['Target']:.2f}.")
            names = ", ".join(top_contributors(df, "Fiber %")["Ingredient"].astype(str).tolist())
            fixes.append(f"Reduce high-fiber ingredients such as {names}, or relax the fiber limit.")
        else:
            issues.append(f"{r['Nutrient']} is blocked: best possible value under current bounds is {r['Max achievable under current bounds']:.2f}, below the target of {r['Target']:.2f}.")
            names = ", ".join(top_contributors(df, r["Nutrient"])["Ingredient"].astype(str).tolist())
            fixes.append(f"Increase Max % for strong {r['Nutrient']} sources such as {names}, add a richer ingredient, or lower the target.")
    if blocked.empty and not issues:
        summary.append("No major mathematical blocker found under the current bounds.")
    elif not blocked.empty:
        primary = blocked.iloc[0]["Nutrient"]
        mapping = {"CP %":"Main bottleneck: protein ceiling is too low.","ME kcal/kg":"Main bottleneck: energy density is too low.","Lys %":"Main bottleneck: lysine target is too high for the current ingredients.","Met %":"Main bottleneck: methionine target is too high for the current ingredients.","Ca %":"Main bottleneck: calcium supply cannot reach the target under current bounds.","AvP %":"Main bottleneck: available phosphorus supply is too low.","Fiber % (max)":"Main bottleneck: the formula is forced to be too fibrous."}
        summary.append(mapping.get(primary, "Main bottleneck: ingredient bounds conflict with the targets."))
    return {"summary":summary,"issues":issues,"fixes":list(dict.fromkeys(fixes)),"capacity_table":cap}

def optimize_formula(df, targets):
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not installed. Run: python -m pip install scipy")
    data = df.copy()
    c = data["Active Price/kg"].to_numpy()
    A_ub, b_ub = [], []
    A_ub.append(-data["CP %"].to_numpy()); b_ub.append(-targets["CP %"] * 100)
    A_ub.append(-data["ME kcal/kg"].to_numpy()); b_ub.append(-targets["ME kcal/kg"] * 100)
    A_ub.append(-data["Lys %"].to_numpy()); b_ub.append(-targets["Lys %"] * 100)
    A_ub.append(-data["Met %"].to_numpy()); b_ub.append(-targets["Met %"] * 100)
    A_ub.append(-data["Ca %"].to_numpy()); b_ub.append(-targets["Ca %"] * 100)
    A_ub.append(-data["AvP %"].to_numpy()); b_ub.append(-targets["AvP %"] * 100)
    A_ub.append(data["Fiber %"].to_numpy()); b_ub.append(targets["Max Fiber %"] * 100)
    A_eq = [np.ones(len(data))]
    b_eq = [targets["Total inclusion"]]
    bounds = list(zip(data["Min %"].to_numpy(), data["Max %"].to_numpy()))
    return linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")

def make_excel_download(ingredients, market, targets_df, nutrient_table, diagnostic_table):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        ingredients.to_excel(writer, index=False, sheet_name="Ingredient_DB")
        market.to_excel(writer, index=False, sheet_name="Market_Prices")
        targets_df.to_excel(writer, index=False, sheet_name="Targets")
        nutrient_table.to_excel(writer, index=False, sheet_name="Dashboard_Metrics")
        diagnostic_table.to_excel(writer, index=False, sheet_name="Diagnostics")
    output.seek(0)
    return output

def reset_data():
    st.session_state.market_df = MARKET_DEFAULTS.copy()
    st.session_state.ingredients_df = INGREDIENT_DEFAULTS.copy()

if "market_df" not in st.session_state: reset_data()
if "selected_profile" not in st.session_state: st.session_state.selected_profile = "Layer Peak"
if "custom_targets" not in st.session_state: st.session_state.custom_targets = PROFILE_DEFAULTS["Custom"].copy()
if "weights" not in st.session_state: st.session_state.weights = WEIGHTS_DEFAULT.copy()

with st.sidebar:
    st.title("SmartFeed Pro")
    st.caption("Merged production version")
    st.session_state.selected_profile = st.selectbox("Diet profile", list(PROFILE_DEFAULTS.keys()), index=list(PROFILE_DEFAULTS.keys()).index(st.session_state.selected_profile))
    if st.button("Load selected profile defaults"):
        if st.session_state.selected_profile != "Custom":
            st.session_state.custom_targets = PROFILE_DEFAULTS[st.session_state.selected_profile].copy()
        st.rerun()
    if st.button("Reset demo data"):
        reset_data()
        st.rerun()
    st.markdown("### Guidance")
    st.markdown("""
- Use **Ingredient DB** to manage your ingredient system.
- Use **Targets** to set the diet goal.
- Use **Diagnostic** before optimization.
- Only run **Optimizer** after fixing blockers.
- Export the workbook snapshot when you want a saved version.
""")

tabs = st.tabs(["Overview","Ingredient DB","Market Prices","Targets","Dashboard","Diagnostic","Optimizer","Export"])

with tabs[0]:
    render_overview("SmartFeed Pro")

with tabs[1]:
    st.subheader("Ingredient database and formulation input")
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("Add blank ingredient row"):
            new_row = {col:0 for col in NUMERIC_COLS}
            new_row.update({"Ingredient":"","Category":"","Price Mode":"Manual","Benchmark Commodity":"","Notes":"","Source URL":""})
            st.session_state.ingredients_df = pd.concat([st.session_state.ingredients_df, pd.DataFrame([new_row])], ignore_index=True)
            st.rerun()
    with col_b:
        remove_idx = st.number_input("Delete row number", min_value=1, max_value=max(1, len(st.session_state.ingredients_df)), value=len(st.session_state.ingredients_df), step=1)
        if st.button("Delete selected row"):
            if len(st.session_state.ingredients_df) > 1:
                st.session_state.ingredients_df = st.session_state.ingredients_df.drop(st.session_state.ingredients_df.index[int(remove_idx)-1]).reset_index(drop=True)
                st.rerun()
            else:
                st.warning("At least one row must remain.")
    commodity_options = [""] + prepare_market_df(st.session_state.market_df)["Commodity"].tolist()
    edited = st.data_editor(
        st.session_state.ingredients_df, num_rows="dynamic", use_container_width=True,
        column_config={
            "Price Mode": st.column_config.SelectboxColumn(options=["Manual","Benchmark"]),
            "Benchmark Commodity": st.column_config.SelectboxColumn(options=commodity_options),
            "Gut Score 0-10": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=0.1),
            "Digestibility 0-10": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, step=0.1),
            "Inclusion %": st.column_config.NumberColumn(step=0.1),
            "Min %": st.column_config.NumberColumn(step=0.1),
            "Max %": st.column_config.NumberColumn(step=0.1),
        },
        hide_index=True, key="ingredient_editor")
    st.session_state.ingredients_df = edited.copy()
    st.caption("Tip: after adding a new ingredient, fill nutrient values, price mode, min/max bounds, and optional gut/digestibility ratings.")

with tabs[2]:
    st.subheader("Market benchmark prices")
    market_edited = st.data_editor(
        st.session_state.market_df, num_rows="dynamic", use_container_width=True,
        column_config={"Quote Unit": st.column_config.SelectboxColumn(options=["USD/kg","USD/metric ton","USD/lb","USD/bushel corn","USD/bushel soybeans"])},
        hide_index=True, key="market_editor")
    st.session_state.market_df = market_edited.copy()
    market_view = prepare_market_df(st.session_state.market_df)
    st.dataframe(market_view[["Commodity","Raw Market Quote","Quote Unit","Benchmark USD/kg","Last Checked"]], use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Nutrient targets and scoring settings")
    profile_source = st.session_state.custom_targets.copy() if st.session_state.selected_profile == "Custom" else PROFILE_DEFAULTS[st.session_state.selected_profile].copy()
    if st.button("Copy selected profile into editable targets"):
        st.session_state.custom_targets = profile_source.copy()
        st.session_state.selected_profile = "Custom"
        st.rerun()
    targets = st.session_state.custom_targets.copy() if st.session_state.selected_profile == "Custom" else profile_source.copy()
    cols = st.columns(3)
    keys = list(targets.keys())
    for i, k in enumerate(keys):
        with cols[i % 3]:
            step_value = 0.01 if "%" in k or "Cost" in k else 1.0
            targets[k] = st.number_input(k, value=float(targets[k]), step=step_value, key=f"target_{k}")
    st.session_state.custom_targets = targets.copy()
    st.markdown("#### Feed-efficiency scoring weights")
    wcols = st.columns(4)
    for i, k in enumerate(st.session_state.weights.keys()):
        with wcols[i]:
            st.session_state.weights[k] = st.number_input(k, min_value=0.0, max_value=1.0, value=float(st.session_state.weights[k]), step=0.05, key=f"weight_{k}")

with tabs[4]:
    market_now = prepare_market_df(st.session_state.market_df)
    ing_now = prepare_ingredient_df(st.session_state.ingredients_df, market_now)
    targets_now = current_targets(st.session_state.selected_profile, st.session_state.custom_targets)
    metrics = compute_metrics(ing_now, targets_now, st.session_state.weights)
    st.subheader("Dashboard")
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Cost/kg", f"${metrics['cost']:.4f}")
    k2.metric("Nutrition adequacy", f"{metrics['nutrition_adequacy']:.1f}%")
    k3.metric("Gut score", f"{metrics['gut']:.2f}/10")
    k4.metric("Feed efficiency index", f"{metrics['feed_eff']:.1f}/100")
    k5.metric("Total inclusion", f"{metrics['total']:.2f}%")
    left,right = st.columns([1.1,0.9])
    with left:
        st.markdown("#### Nutrient check")
        st.dataframe(metrics["nutrient_table"], use_container_width=True, hide_index=True)
    with right:
        mix = ing_now[ing_now["Inclusion %"] > 0][["Ingredient","Inclusion %"]].sort_values("Inclusion %", ascending=False)
        if not mix.empty:
            chart = alt.Chart(mix).mark_arc().encode(theta="Inclusion %:Q", color="Ingredient:N", tooltip=["Ingredient","Inclusion %"])
            st.altair_chart(chart, use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        cost_df = ing_now.copy()
        cost_df["Cost Contribution"] = cost_df["Inclusion %"] / 100.0 * cost_df["Active Price/kg"]
        cost_df = cost_df[cost_df["Cost Contribution"] > 0].sort_values("Cost Contribution", ascending=False)
        bar = alt.Chart(cost_df).mark_bar().encode(x=alt.X("Ingredient:N", sort="-y"), y="Cost Contribution:Q", tooltip=["Ingredient","Active Price/kg","Inclusion %","Cost Contribution"])
        st.markdown("#### Cost contribution")
        st.altair_chart(bar, use_container_width=True)
    with c2:
        score_df = pd.DataFrame([{"Component":"Nutrition adequacy","Score":metrics["nutrition_adequacy"]},{"Component":"Gut score x10","Score":metrics["gut"]*10},{"Component":"Cost score","Score":metrics["cost_score"]},{"Component":"E:P balance","Score":metrics["ep_balance"]}])
        bar2 = alt.Chart(score_df).mark_bar().encode(x="Component:N", y="Score:Q", tooltip=["Component","Score"])
        st.markdown("#### Score components")
        st.altair_chart(bar2, use_container_width=True)

with tabs[5]:
    st.subheader("Diagnostic intelligence")
    market_now = prepare_market_df(st.session_state.market_df)
    ing_now = prepare_ingredient_df(st.session_state.ingredients_df, market_now)
    targets_now = current_targets(st.session_state.selected_profile, st.session_state.custom_targets)
    diagnosis = diagnose_feasibility(ing_now, targets_now)
    for s in diagnosis["summary"]:
        st.info(s)
    if diagnosis["issues"]:
        st.markdown("#### What is blocking the formula")
        for item in diagnosis["issues"]:
            st.error(item)
    else:
        st.success("No major mathematical blocker found under the current bounds.")
    if diagnosis["fixes"]:
        st.markdown("#### What to change first")
        for i, fix in enumerate(diagnosis["fixes"], start=1):
            st.write(f"{i}. {fix}")
    st.markdown("#### Capacity table")
    st.dataframe(diagnosis["capacity_table"], use_container_width=True, hide_index=True)
    blocked = diagnosis["capacity_table"][diagnosis["capacity_table"]["Status"] == "BLOCKED"].copy()
    if not blocked.empty:
        blocked_chart = alt.Chart(blocked).mark_bar().encode(x="Nutrient:N", y="Gap to target:Q", tooltip=["Nutrient","Target","Max achievable under current bounds","Gap to target"])
        st.markdown("#### Gap to target")
        st.altair_chart(blocked_chart, use_container_width=True)

with tabs[6]:
    st.subheader("Least-cost optimizer")
    st.write("Run this only after checking the Diagnostic tab.")
    market_now = prepare_market_df(st.session_state.market_df)
    ing_now = prepare_ingredient_df(st.session_state.ingredients_df, market_now)
    targets_now = current_targets(st.session_state.selected_profile, st.session_state.custom_targets)
    if not SCIPY_AVAILABLE:
        st.warning("SciPy is not installed, so automatic optimization is disabled. Install it with: python -m pip install scipy")
    else:
        if st.button("Run automatic optimization"):
            try:
                result = optimize_formula(ing_now, targets_now)
                if result.success:
                    opt = ing_now.copy()
                    opt["Inclusion %"] = result.x
                    st.session_state.ingredients_df["Inclusion %"] = result.x
                    opt_metrics = compute_metrics(opt, targets_now, st.session_state.weights)
                    st.success("Optimization solved successfully.")
                    st.dataframe(opt[["Ingredient","Category","Inclusion %","Active Price/kg","CP %","ME kcal/kg","Lys %","Met %","Ca %","AvP %","Fiber %"]], use_container_width=True, hide_index=True)
                    a,b,c,d = st.columns(4)
                    a.metric("Optimized cost/kg", f"${opt_metrics['cost']:.4f}")
                    b.metric("Gut score", f"{opt_metrics['gut']:.2f}/10")
                    c.metric("Feed efficiency index", f"{opt_metrics['feed_eff']:.1f}/100")
                    d.metric("Total inclusion", f"{opt_metrics['total']:.2f}%")
                    st.dataframe(opt_metrics["nutrient_table"], use_container_width=True, hide_index=True)
                else:
                    st.error("Optimization failed.")
                    st.code(str(result.message))
                    diagnosis = diagnose_feasibility(ing_now, targets_now)
                    if diagnosis["issues"]:
                        st.markdown("#### Plain-language diagnosis")
                        for item in diagnosis["issues"][:4]:
                            st.write(f"- {item}")
            except Exception as e:
                st.error(f"Optimizer error: {e}")

with tabs[7]:
    st.subheader("Export current model")
    market_now = prepare_market_df(st.session_state.market_df)
    ing_now = prepare_ingredient_df(st.session_state.ingredients_df, market_now)
    targets_now = current_targets(st.session_state.selected_profile, st.session_state.custom_targets)
    metrics = compute_metrics(ing_now, targets_now, st.session_state.weights)
    diagnosis = diagnose_feasibility(ing_now, targets_now)
    targets_df = pd.DataFrame([{"Metric":k,"Value":v} for k,v in targets_now.items()])
    xlsx = make_excel_download(ing_now, market_now, targets_df, metrics["nutrient_table"], diagnosis["capacity_table"])
    st.download_button("Download workbook snapshot", data=xlsx, file_name="smartfeed_pro_final_snapshot.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download ingredient DB as CSV", data=ing_now.to_csv(index=False).encode("utf-8"), file_name="ingredient_db.csv", mime="text/csv")
