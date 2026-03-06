import streamlit as st
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
import io
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Integrated Fuzzy MCDM Models",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
<style>
:root {
    --primary: #1f77b4;
    --secondary: #2ca02c;
    --accent: #ff6b6b;
    --background: #f8f9fa;
    --card-bg: #ffffff;
    --text: #262730;
    --border-radius: 12px;
    --shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
}

.main-header {
    font-size: 2.6rem;
    color: var(--primary);
    text-align: center;
    padding: 1.2rem 0;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-header {
    font-size: 1.5rem;
    color: var(--primary);
    border-left: 5px solid var(--secondary);
    padding-left: 1rem;
    margin: 1.5rem 0 1rem 0;
    font-weight: 600;
}

.panel {
    background-color: var(--card-bg);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
    border: 1px solid #e0e0e0;
}

.result-table {
    background-color: var(--card-bg);
    border-radius: var(--border-radius);
    padding: 1.2rem;
    box-shadow: var(--shadow);
    margin: 1rem 0;
}

.metric-card {
    background: linear-gradient(135deg, var(--card-bg), #f7f9fc);
    border-radius: var(--border-radius);
    padding: 1.2rem;
    box-shadow: var(--shadow);
    text-align: center;
    border: 1px solid #e8e8e8;
}

.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--primary);
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.9rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.optimization-formulation {
    background-color: #f8f9fa;
    border-left: 4px solid var(--primary);
    padding: 1.2rem;
    border-radius: 8px;
    margin: 1rem 0;
    font-family: 'Courier New', monospace;
    font-size: 0.92rem;
    line-height: 1.6;
}

.instruction-box {
    background-color: #f0f7ff;
    border-left: 4px solid var(--primary);
    padding: 1rem 1.2rem;
    border-radius: 4px;
    margin: 1rem 0;
}

.footer {
    text-align: center;
    margin-top: 2rem;
    padding: 1rem;
    color: #666;
    font-size: 0.9rem;
    border-top: 1px solid #eaeaea;
}

.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 40px;
    font-weight: 600;
    width: 100%;
}
</style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# TRIANGULAR FUZZY SCALE (USER-PROVIDED)
# =========================================================
# Format: (l, m, u)
LINGUISTIC_SCALE = {
    "AL": (1.00, 1.50, 2.50),   # Absolutely Low
    "VL": (1.50, 2.50, 3.50),   # Very Low
    "L":  (2.50, 3.50, 4.50),   # Low
    "ML": (3.50, 4.50, 5.50),   # Medium Low
    "E":  (4.50, 5.50, 6.50),   # Equal
    "MH": (5.50, 6.50, 7.50),   # Medium High
    "H":  (6.50, 7.50, 8.50),   # High
    "VH": (7.50, 8.50, 9.50),   # Very High
    "AH": (8.50, 9.00, 10.00),  # Absolutely High
}

LINGUISTIC_LABELS = {
    "AL": "Absolutely Low",
    "VL": "Very Low",
    "L": "Low",
    "ML": "Medium Low",
    "E": "Equal",
    "MH": "Medium High",
    "H": "High",
    "VH": "Very High",
    "AH": "Absolutely High",
}

LINGUISTIC_OPTIONS = list(LINGUISTIC_SCALE.keys())


# =========================================================
# SHARED FUZZY FUNCTIONS
# =========================================================
def trig_geom_component(values, weights):
    """
    Trigonometric aggregation component matching the user's Excel formula:
    aggregated = sum(values) * (2/pi) * acos( product( cos(pi * v/sum(values) / 2) ** w ) )
    """
    s = sum(values)
    if s == 0:
        return 0.0

    prod = 1.0
    for v, w in zip(values, weights):
        ratio = v / s
        term = np.cos(np.pi * ratio / 2) ** w
        prod *= term

    prod = np.clip(prod, -1.0, 1.0)
    return s * (2 / np.pi) * np.arccos(prod)


def aggregate_tfn(tfn_list, weights):
    ls = [t[0] for t in tfn_list]
    ms = [t[1] for t in tfn_list]
    us = [t[2] for t in tfn_list]
    l = trig_geom_component(ls, weights)
    m = trig_geom_component(ms, weights)
    u = trig_geom_component(us, weights)
    return (l, m, u)


def defuzz_tfn(tfn):
    """Graded mean integration representation for TFN."""
    return (tfn[0] + 4 * tfn[1] + tfn[2]) / 6.0


def crisp_to_tfn(x, alpha=0.05):
    x = float(x)
    return (x, x + alpha, x + 2 * alpha)


def format_tfn(tfn):
    return f"({tfn[0]:.4f}, {tfn[1]:.4f}, {tfn[2]:.4f})"


# =========================================================
# OPA MODULE (TRIANGULAR FUZZY OPA)
# =========================================================
def solve_fuzzy_opa(coeff_list, n):
    prob = LpProblem("Triangular_Fuzzy_OPA", LpMaximize)

    w_l = [LpVariable(f"w_l_{i}", lowBound=0) for i in range(n)]
    w_m = [LpVariable(f"w_m_{i}", lowBound=0) for i in range(n)]
    w_u = [LpVariable(f"w_u_{i}", lowBound=0) for i in range(n)]

    psi_l = LpVariable("psi_l", lowBound=0)
    psi_m = LpVariable("psi_m", lowBound=0)
    psi_u = LpVariable("psi_u", lowBound=0)

    prob += (psi_l + psi_m + psi_u) / 3

    for i in range(n):
        prob += w_l[i] <= w_m[i]
        prob += w_m[i] <= w_u[i]

    # TFN normalization
    prob += lpSum(w_l) == 0.9
    prob += lpSum(w_m) == 1.0
    prob += lpSum(w_u) == 1.1

    for a in range(n - 1):
        prob += coeff_list[a][0] * (w_l[a] - w_u[a + 1]) >= psi_l
        prob += coeff_list[a][1] * (w_m[a] - w_m[a + 1]) >= psi_m
        prob += coeff_list[a][2] * (w_u[a] - w_l[a + 1]) >= psi_u

    prob += coeff_list[n - 1][0] * w_l[n - 1] >= psi_l
    prob += coeff_list[n - 1][1] * w_m[n - 1] >= psi_m
    prob += coeff_list[n - 1][2] * w_u[n - 1] >= psi_u

    status = prob.solve()
    if status != 1:
        st.error("Optimization failed. The model may be infeasible. Please check your inputs.")
        return None, None

    weights = []
    for i in range(n):
        weights.append((
            max(0, value(w_l[i])),
            max(0, value(w_m[i])),
            max(0, value(w_u[i]))
        ))

    psi = (
        max(0, value(psi_l)),
        max(0, value(psi_m)),
        max(0, value(psi_u))
    )
    return weights, psi


def display_linguistic_scale():
    scale_df = pd.DataFrame([
        {"Code": k, "Linguistic Attribute": LINGUISTIC_LABELS[k], "l": v[0], "m": v[1], "u": v[2]}
        for k, v in LINGUISTIC_SCALE.items()
    ])
    st.dataframe(scale_df, use_container_width=True, hide_index=True)


def create_opa_word_document(criteria, theta, defuzz_values, coeff, ranked_criteria, weights, psi, num_experts, expert_weights):
    doc = Document()
    title = doc.add_heading('Triangular Fuzzy OPA Results - Multiple Experts', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(f'This document contains the results of the Triangular Fuzzy OPA analysis for {num_experts} experts.')
    doc.add_paragraph('')

    doc.add_heading('Expert Weights', level=1)
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    hdr[0].text = 'Expert'
    hdr[1].text = 'Weight'
    for e in range(num_experts):
        row = table.add_row().cells
        row[0].text = f'Expert {e+1}'
        row[1].text = f'{expert_weights[e]:.4f}'

    doc.add_paragraph('')
    doc.add_heading('Aggregated Triangular Fuzzy Importance (Theta)', level=1)
    table = doc.add_table(rows=1, cols=5)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    hdr[0].text = 'Criterion'
    hdr[1].text = 'l'
    hdr[2].text = 'm'
    hdr[3].text = 'u'
    hdr[4].text = 'Defuzzified'
    for i, crit in enumerate(criteria):
        row = table.add_row().cells
        row[0].text = crit
        row[1].text = f'{theta[i][0]:.4f}'
        row[2].text = f'{theta[i][1]:.4f}'
        row[3].text = f'{theta[i][2]:.4f}'
        row[4].text = f'{defuzz_values[i]:.4f}'

    doc.add_paragraph('')
    doc.add_heading('Coefficients for Fuzzy OPA', level=1)
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    hdr[0].text = 'Criterion'
    hdr[1].text = 'Coeff l'
    hdr[2].text = 'Coeff m'
    hdr[3].text = 'Coeff u'
    for i, crit in enumerate(criteria):
        row = table.add_row().cells
        row[0].text = crit
        row[1].text = f'{coeff[i][0]:.4f}'
        row[2].text = f'{coeff[i][1]:.4f}'
        row[3].text = f'{coeff[i][2]:.4f}'

    doc.add_paragraph('')
    doc.add_heading('Ranked Criteria', level=1)
    table = doc.add_table(rows=1, cols=6)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    hdr[0].text = 'Rank'
    hdr[1].text = 'Criterion'
    hdr[2].text = 'Weight l'
    hdr[3].text = 'Weight m'
    hdr[4].text = 'Weight u'
    hdr[5].text = 'Defuzzified'
    for rank, (crit_idx, w) in enumerate(ranked_criteria):
        row = table.add_row().cells
        row[0].text = str(rank + 1)
        row[1].text = criteria[crit_idx]
        row[2].text = f'{w[0]:.4f}'
        row[3].text = f'{w[1]:.4f}'
        row[4].text = f'{w[2]:.4f}'
        row[5].text = f'{defuzz_tfn(w):.4f}'

    doc.add_paragraph('')
    doc.add_heading('Psi Value', level=1)
    doc.add_paragraph(f'Psi: ({psi[0]:.4f}, {psi[1]:.4f}, {psi[2]:.4f})')
    doc.add_paragraph(f'Defuzzified Psi: {defuzz_tfn(psi):.4f}')

    doc_bytes = io.BytesIO()
    doc.save(doc_bytes)
    doc_bytes.seek(0)
    return doc_bytes


def display_optimization_formulation(coeff_list, criteria_names):
    st.markdown('<div class="optimization-formulation">', unsafe_allow_html=True)
    st.markdown(
        """
        <strong>Triangular Fuzzy OPA Formulation</strong><br><br>
        Objective:<br>
        Maximize: (Ψ_l + Ψ_m + Ψ_u) / 3<br><br>
        Constraints:<br>
        1. Triangular ordering: w_lᵢ ≤ w_mᵢ ≤ w_uᵢ<br>
        2. Normalization: Σw_lᵢ = 0.9, Σw_mᵢ = 1.0, Σw_uᵢ = 1.1<br>
        3. Adjacent ranked criteria:<br>
        &nbsp;&nbsp;θ_lᵃ(w_lᵃ - w_uᵃ⁺¹) ≥ Ψ_l<br>
        &nbsp;&nbsp;θ_mᵃ(w_mᵃ - w_mᵃ⁺¹) ≥ Ψ_m<br>
        &nbsp;&nbsp;θ_uᵃ(w_uᵃ - w_lᵃ⁺¹) ≥ Ψ_u<br>
        4. Last criterion:<br>
        &nbsp;&nbsp;θ_lⁿw_lⁿ ≥ Ψ_l, θ_mⁿw_mⁿ ≥ Ψ_m, θ_uⁿw_uⁿ ≥ Ψ_u
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    coeff_df = pd.DataFrame({
        'Criterion': criteria_names,
        'θ_l': [f'{c[0]:.4f}' for c in coeff_list],
        'θ_m': [f'{c[1]:.4f}' for c in coeff_list],
        'θ_u': [f'{c[2]:.4f}' for c in coeff_list],
    })
    st.dataframe(coeff_df, use_container_width=True, hide_index=True)


def opa_model():
    st.markdown('<h1 class="main-header">Triangular Fuzzy OPA Analysis with Multiple Experts</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem; color: #555;">
        This application implements a <b>Triangular Fuzzy OPA</b> model using your updated linguistic scale.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if 'opa_criteria' not in st.session_state:
        st.session_state.opa_criteria = [f'Criterion {i+1}' for i in range(5)]
    if 'opa_num_criteria' not in st.session_state:
        st.session_state.opa_num_criteria = 5
    if 'opa_num_experts' not in st.session_state:
        st.session_state.opa_num_experts = 2
    if 'opa_results_calculated' not in st.session_state:
        st.session_state.opa_results_calculated = False
    if 'opa_expert_data' not in st.session_state:
        st.session_state.opa_expert_data = {}
    if 'opa_expert_weights' not in st.session_state:
        st.session_state.opa_expert_weights = [0.5, 0.5]

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Step 1: Define Criteria and Experts</h2>', unsafe_allow_html=True)

        num_experts = st.number_input('Number of experts', min_value=1, max_value=15, value=st.session_state.opa_num_experts, step=1)
        num_criteria = st.number_input('Number of criteria', min_value=3, max_value=25, value=st.session_state.opa_num_criteria, step=1)

        if num_criteria != st.session_state.opa_num_criteria or num_experts != st.session_state.opa_num_experts:
            st.session_state.opa_num_criteria = num_criteria
            st.session_state.opa_num_experts = num_experts
            if len(st.session_state.opa_criteria) < num_criteria:
                for i in range(len(st.session_state.opa_criteria), num_criteria):
                    st.session_state.opa_criteria.append(f'Criterion {i+1}')
            else:
                st.session_state.opa_criteria = st.session_state.opa_criteria[:num_criteria]
            st.session_state.opa_expert_data = {}
            st.session_state.opa_expert_weights = [1.0 / num_experts] * num_experts
            st.session_state.opa_results_calculated = False

        criteria = []
        for i in range(num_criteria):
            c = st.text_input(f'Criterion {i+1}', value=st.session_state.opa_criteria[i], key=f'opa_criterion_{i}')
            criteria.append(c)
        st.session_state.opa_criteria = criteria

        st.markdown('<h3>Expert Weights</h3>', unsafe_allow_html=True)
        expert_weights = []
        for e in range(num_experts):
            w = st.number_input(
                f'Weight for Expert {e+1}',
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.opa_expert_weights[e]) if e < len(st.session_state.opa_expert_weights) else 1.0 / num_experts,
                step=0.0001,
                format='%.4f',
                key=f'opa_expert_w_{e}'
            )
            expert_weights.append(w)

        sum_w = sum(expert_weights)
        st.write(f'**Sum of expert weights: {sum_w:.4f}**')
        if abs(sum_w - 1.0) > 1e-6:
            st.error('Expert weights must sum to 1.00.')
            st.session_state.opa_weights_valid = False
        else:
            st.session_state.opa_expert_weights = expert_weights
            st.session_state.opa_weights_valid = True
            st.success('Expert weights are valid.')

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Expert Assessments</h2>', unsafe_allow_html=True)
        st.markdown('<div class="instruction-box"><b>Updated linguistic scale</b></div>', unsafe_allow_html=True)
        display_linguistic_scale()

        tabs = st.tabs([f'Expert {e+1}' for e in range(num_experts)])
        for e, tab in enumerate(tabs):
            with tab:
                if f'expert_{e}' not in st.session_state.opa_expert_data:
                    st.session_state.opa_expert_data[f'expert_{e}'] = ['E'] * num_criteria

                df = pd.DataFrame({
                    'Criterion': criteria,
                    'Rating': st.session_state.opa_expert_data[f'expert_{e}']
                })
                edited_df = st.data_editor(
                    df,
                    column_config={
                        'Rating': st.column_config.SelectboxColumn('Rating', options=LINGUISTIC_OPTIONS, required=True)
                    },
                    use_container_width=True,
                    key=f'opa_data_editor_expert_{e}'
                )
                st.session_state.opa_expert_data[f'expert_{e}'] = edited_df['Rating'].tolist()

        all_ratings_available = all(f'expert_{e}' in st.session_state.opa_expert_data for e in range(num_experts))
        weights_valid = getattr(st.session_state, 'opa_weights_valid', False)
        disabled = not (all_ratings_available and weights_valid)

        if st.button('Calculate Weights', disabled=disabled, use_container_width=True):
            theta = []
            defuzz_values = []
            for j in range(num_criteria):
                tfn_list = [LINGUISTIC_SCALE[st.session_state.opa_expert_data[f'expert_{e}'][j]] for e in range(num_experts)]
                aggregated = aggregate_tfn(tfn_list, st.session_state.opa_expert_weights)
                theta.append(aggregated)
                defuzz_values.append(defuzz_tfn(aggregated))

            min_l = min(t[0] for t in theta if t[0] > 0) if any(t[0] > 0 for t in theta) else 1.0

            coeff = []
            for t in theta:
                if min(t) <= 0:
                    coeff.append((0, 0, 0))
                else:
                    coeff.append((min_l / t[2], min_l / t[1], min_l / t[0]))

            sorted_indices = np.argsort(defuzz_values)[::-1]
            coeff_sorted = [coeff[idx] for idx in sorted_indices]
            weights_sorted, psi = solve_fuzzy_opa(coeff_sorted, num_criteria)

            if weights_sorted is not None:
                weights = [None] * num_criteria
                for rank, idx in enumerate(sorted_indices):
                    weights[idx] = weights_sorted[rank]
                ranked_criteria = [(sorted_indices[k], weights_sorted[k]) for k in range(num_criteria)]

                st.session_state.opa_theta = theta
                st.session_state.opa_defuzz_values = defuzz_values
                st.session_state.opa_coeff = coeff
                st.session_state.opa_ranked_criteria = ranked_criteria
                st.session_state.opa_weights = weights
                st.session_state.opa_psi = psi
                st.session_state.opa_results_calculated = True
                st.success('Calculation completed successfully.')

        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.opa_results_calculated:
        st.markdown('<h2 class="section-header">Results</h2>', unsafe_allow_html=True)

        st.markdown('<div class="result-table">', unsafe_allow_html=True)
        st.subheader('Normalized Expert Weights')
        df_expert_w = pd.DataFrame({
            'Expert': [f'Expert {e+1}' for e in range(st.session_state.opa_num_experts)],
            'Weight': st.session_state.opa_expert_weights
        })
        st.dataframe(df_expert_w, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="result-table">', unsafe_allow_html=True)
        st.subheader('Aggregated Triangular Fuzzy Importance (Theta)')
        df_theta = pd.DataFrame({
            'Criterion': st.session_state.opa_criteria,
            'l': [t[0] for t in st.session_state.opa_theta],
            'm': [t[1] for t in st.session_state.opa_theta],
            'u': [t[2] for t in st.session_state.opa_theta],
            'Defuzzified': st.session_state.opa_defuzz_values,
        })
        st.dataframe(df_theta, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="result-table">', unsafe_allow_html=True)
        st.subheader('Coefficients for Fuzzy OPA')
        df_coeff = pd.DataFrame({
            'Criterion': st.session_state.opa_criteria,
            'Coeff l': [c[0] for c in st.session_state.opa_coeff],
            'Coeff m': [c[1] for c in st.session_state.opa_coeff],
            'Coeff u': [c[2] for c in st.session_state.opa_coeff],
        })
        st.dataframe(df_coeff, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        sorted_indices = np.argsort(st.session_state.opa_defuzz_values)[::-1]
        coeff_sorted = [st.session_state.opa_coeff[idx] for idx in sorted_indices]
        criteria_sorted = [st.session_state.opa_criteria[idx] for idx in sorted_indices]
        display_optimization_formulation(coeff_sorted, criteria_sorted)

        st.markdown('<div class="result-table">', unsafe_allow_html=True)
        st.subheader('Triangular Fuzzy Weights')
        df_weights = pd.DataFrame({
            'Criterion': st.session_state.opa_criteria,
            'l': [w[0] for w in st.session_state.opa_weights],
            'm': [w[1] for w in st.session_state.opa_weights],
            'u': [w[2] for w in st.session_state.opa_weights],
            'Defuzzified': [defuzz_tfn(w) for w in st.session_state.opa_weights],
        })
        st.dataframe(df_weights, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="result-table">', unsafe_allow_html=True)
        st.subheader('Ranked Criteria and Weights')
        df_ranked = pd.DataFrame({
            'Rank': list(range(1, st.session_state.opa_num_criteria + 1)),
            'Criterion': [st.session_state.opa_criteria[idx] for idx, _ in st.session_state.opa_ranked_criteria],
            'Weight l': [w[0] for _, w in st.session_state.opa_ranked_criteria],
            'Weight m': [w[1] for _, w in st.session_state.opa_ranked_criteria],
            'Weight u': [w[2] for _, w in st.session_state.opa_ranked_criteria],
            'Defuzzified': [defuzz_tfn(w) for _, w in st.session_state.opa_ranked_criteria],
        })
        st.dataframe(df_ranked, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Psi (l, m, u)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{st.session_state.opa_psi[0]:.4f}, {st.session_state.opa_psi[1]:.4f}, {st.session_state.opa_psi[2]:.4f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Defuzzified Psi</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{defuzz_tfn(st.session_state.opa_psi):.4f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        doc_bytes = create_opa_word_document(
            st.session_state.opa_criteria,
            st.session_state.opa_theta,
            st.session_state.opa_defuzz_values,
            st.session_state.opa_coeff,
            st.session_state.opa_ranked_criteria,
            st.session_state.opa_weights,
            st.session_state.opa_psi,
            st.session_state.opa_num_experts,
            st.session_state.opa_expert_weights,
        )
        st.download_button(
            label='Export Results to Word',
            data=doc_bytes,
            file_name='Triangular_Fuzzy_OPA_Results.docx',
            mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            use_container_width=True,
        )

    with st.expander('Learn more about the Triangular Fuzzy OPA Method'):
        st.markdown(
            """
            1. Experts provide linguistic assessments.
            2. Linguistic terms are converted into triangular fuzzy numbers (TFNs).
            3. Expert opinions are aggregated using the trigonometric fuzzy weighted geometric operator.
            4. Aggregated TFNs are defuzzified and ranked.
            5. A triangular fuzzy linear programming model computes final weights.
            """
        )
        display_linguistic_scale()


# =========================================================
# TRUST MODULE (TRIANGULAR FUZZY TRUST)
# =========================================================
def create_trust_word_document(all_data):
    doc = Document()
    title = doc.add_heading('Triangular Fuzzy TRUST Method Results', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('Problem Setup', level=1)
    table = doc.add_table(rows=5, cols=2)
    table.style = 'Table Grid'
    setup_data = [
        ('Number of Alternatives', all_data['n_alternatives']),
        ('Number of Criteria', all_data['n_criteria']),
        ('Number of Experts', all_data['n_experts']),
        ('Alpha Parameters', f"α₁: {all_data['alpha'][0]}, α₂: {all_data['alpha'][1]}, α₃: {all_data['alpha'][2]}, α₄: {all_data['alpha'][3]}"),
        ('Beta Parameter', all_data['beta']),
    ]
    for i, (label, value) in enumerate(setup_data):
        row = table.rows[i].cells
        row[0].text = label
        row[1].text = str(value)

    doc.add_paragraph('')
    doc.add_heading('Final Ranking Results', level=1)
    table = doc.add_table(rows=1, cols=5)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    hdr[0].text = 'Rank'
    hdr[1].text = 'Alternative'
    hdr[2].text = '∑Pik'
    hdr[3].text = '∑Hik'
    hdr[4].text = 'L Score'
    for _, row_data in all_data['final_results'].iterrows():
        row = table.add_row().cells
        row[0].text = str(row_data['Rank'])
        row[1].text = row_data['Alternative']
        row[2].text = f"{row_data['∑Pik']:.4f}"
        row[3].text = f"{row_data['∑Hik']:.4f}"
        row[4].text = f"{row_data['L Score']:.4f}"

    doc.add_paragraph('')
    doc.add_paragraph(f"Best Alternative: {all_data['best_alternative']} with score: {all_data['best_score']:.4f}")

    doc_bytes = io.BytesIO()
    doc.save(doc_bytes)
    doc_bytes.seek(0)
    return doc_bytes


def trust_model():
    st.markdown('<h1 class="main-header">Triangular Fuzzy TRUST Method</h1>', unsafe_allow_html=True)
    st.write('Enhanced version with soft/hard criteria handling, expert aggregation, and your updated triangular linguistic scale.')

    with st.expander('Learn more about the Triangular Fuzzy TRUST Method', expanded=False):
        st.markdown(
            """
            This version replaces trapezoidal fuzzy sets with triangular fuzzy numbers (TFNs).

            **Key changes:**
            - Soft criteria use your custom 9-level linguistic TFN scale.
            - Hard criteria are transformed to TFNs using a small spread.
            - Aggregation uses a trigonometric triangular fuzzy weighted geometric operator.
            - Defuzzification uses the centroid of TFNs.
            """
        )
        display_linguistic_scale()

    if 'trust_step' not in st.session_state:
        st.session_state.trust_step = 1
    if 'trust_data' not in st.session_state:
        st.session_state.trust_data = {}

    steps = [
        'Problem Setup', 'Criteria Setup', 'Expert Weights', 'Data Collection',
        'Build Decision Matrix', 'Criteria Information', 'Constraint Values', 'Results'
    ]
    current_step = st.session_state.trust_step - 1
    st.progress(current_step / (len(steps) - 1))
    st.write(f'**Current Step: {steps[current_step]}**')

    if st.session_state.trust_step == 1:
        trust_step1_input()
    elif st.session_state.trust_step == 2:
        trust_step2_criteria_setup()
    elif st.session_state.trust_step == 3:
        trust_step3_expert_weights()
    elif st.session_state.trust_step == 4:
        trust_step4_data_collection()
    elif st.session_state.trust_step == 5:
        trust_step5_decision_matrix()
    elif st.session_state.trust_step == 6:
        trust_step6_criteria_info()
    elif st.session_state.trust_step == 7:
        trust_step7_constraints()
    elif st.session_state.trust_step == 8:
        trust_step8_calculations()


def trust_step1_input():
    st.header('Step 1: Problem Setup')
    n_alternatives = st.number_input('Number of alternatives', min_value=2, max_value=20, value=4)
    n_criteria = st.number_input('Number of criteria', min_value=2, max_value=20, value=6)
    n_experts = st.number_input('Number of experts', min_value=1, max_value=10, value=3)

    st.subheader('Normalization Parameters (α)')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        alpha1 = st.number_input('α₁ (Linear Ratio)', min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    with col2:
        alpha2 = st.number_input('α₂ (Linear Sum)', min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    with col3:
        alpha3 = st.number_input('α₃ (Max-Min)', min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    with col4:
        alpha4 = st.number_input('α₄ (Logarithmic)', min_value=0.0, max_value=1.0, value=0.25, step=0.05)

    alpha_sum = alpha1 + alpha2 + alpha3 + alpha4
    if abs(alpha_sum - 1.0) > 1e-6:
        st.warning(f'Alpha values sum to {alpha_sum:.2f}, but should sum to 1.0')

    beta = st.slider('β (Distance Aggregation Parameter)', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    if st.button('Next: Criteria Setup'):
        st.session_state.trust_data['n_alternatives'] = n_alternatives
        st.session_state.trust_data['n_criteria'] = n_criteria
        st.session_state.trust_data['n_experts'] = n_experts
        st.session_state.trust_data['alpha'] = [alpha1, alpha2, alpha3, alpha4]
        st.session_state.trust_data['beta'] = beta
        st.session_state.trust_data['alternatives'] = [f'A{i+1}' for i in range(n_alternatives)]
        st.session_state.trust_data['criteria'] = [f'C{i+1}' for i in range(n_criteria)]
        st.session_state.trust_data['criteria_types'] = ['Soft'] * n_criteria
        st.session_state.trust_data['expert_weights'] = [1.0 / n_experts] * n_experts
        st.session_state.trust_step = 2
        st.rerun()


def trust_step2_criteria_setup():
    st.header('Step 2: Criteria Setup')
    n_criteria = st.session_state.trust_data['n_criteria']
    criteria = st.session_state.trust_data['criteria']

    data = {
        'Criterion': criteria,
        'Type': st.session_state.trust_data['criteria_types'],
        'Description': [''] * n_criteria,
    }
    df = pd.DataFrame(data)
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        column_config={
            'Type': st.column_config.SelectboxColumn('Type', options=['Soft', 'Hard'])
        }
    )

    with st.expander('Linguistic Scale for Soft Criteria'):
        display_linguistic_scale()

    if st.button('Next: Expert Weights'):
        st.session_state.trust_data['criteria_setup'] = edited_df
        st.session_state.trust_data['criteria_types'] = edited_df['Type'].tolist()
        st.session_state.trust_step = 3
        st.rerun()


def trust_step3_expert_weights():
    st.header('Step 3: Expert Weights')
    n_experts = st.session_state.trust_data['n_experts']
    expert_weights = []
    for e in range(n_experts):
        weight = st.number_input(
            f'Weight for Expert {e+1}',
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.trust_data['expert_weights'][e]),
            step=0.0001,
            format='%.4f'
        )
        expert_weights.append(weight)

    weight_sum = sum(expert_weights)
    valid_weights = abs(weight_sum - 1.0) <= 1e-6
    if valid_weights:
        st.success('Expert weights are valid.')
    else:
        st.warning(f'Expert weights sum to {weight_sum:.2f}, but should sum to 1.0')

    if st.button('Next: Data Collection') and valid_weights:
        st.session_state.trust_data['expert_weights'] = expert_weights
        st.session_state.trust_step = 4
        st.rerun()


def trust_step4_data_collection():
    st.header('Step 4: Data Collection')
    n_alternatives = st.session_state.trust_data['n_alternatives']
    n_criteria = st.session_state.trust_data['n_criteria']
    n_experts = st.session_state.trust_data['n_experts']
    alternatives = st.session_state.trust_data['alternatives']
    criteria = st.session_state.trust_data['criteria']
    criteria_types = st.session_state.trust_data['criteria_types']

    if 'expert_data' not in st.session_state.trust_data:
        st.session_state.trust_data['expert_data'] = {}
        for e in range(n_experts):
            st.session_state.trust_data['expert_data'][f'expert_{e}'] = pd.DataFrame(
                [['E'] * n_criteria for _ in range(n_alternatives)],
                columns=criteria,
                index=alternatives,
            )

    if 'hard_data' not in st.session_state.trust_data:
        st.session_state.trust_data['hard_data'] = pd.DataFrame(
            [[0.0] * n_criteria for _ in range(n_alternatives)],
            columns=criteria,
            index=alternatives,
        )

    soft_criteria = [criteria[i] for i, t in enumerate(criteria_types) if str(t).lower() == 'soft']
    hard_criteria = [criteria[i] for i, t in enumerate(criteria_types) if str(t).lower() == 'hard']

    st.write('### Soft Criteria Assessment')
    if soft_criteria:
        expert_tabs = st.tabs([f'Expert {e+1}' for e in range(n_experts)])
        for e, tab in enumerate(expert_tabs):
            with tab:
                current_data = st.session_state.trust_data['expert_data'][f'expert_{e}']
                soft_data = current_data[soft_criteria].copy()
                edited_soft = st.data_editor(
                    soft_data,
                    column_config={
                        col: st.column_config.SelectboxColumn(col, options=LINGUISTIC_OPTIONS, required=True)
                        for col in soft_criteria
                    },
                    use_container_width=True,
                    key=f'trust_expert_{e}_soft'
                )
                st.session_state.trust_data['expert_data'][f'expert_{e}'].update(edited_soft)

    st.write('### Hard Criteria Assessment')
    if hard_criteria:
        current_hard = st.session_state.trust_data['hard_data'][hard_criteria].copy()
        edited_hard = st.data_editor(current_hard, use_container_width=True, key='trust_hard_data')
        st.session_state.trust_data['hard_data'].update(edited_hard)

    if st.button('Next: Build Decision Matrix'):
        st.session_state.trust_step = 5
        st.rerun()


def trust_step5_decision_matrix():
    st.header('Step 5: Build Decision Matrix')
    n_alternatives = st.session_state.trust_data['n_alternatives']
    n_criteria = st.session_state.trust_data['n_criteria']
    n_experts = st.session_state.trust_data['n_experts']
    alternatives = st.session_state.trust_data['alternatives']
    criteria = st.session_state.trust_data['criteria']
    criteria_types = st.session_state.trust_data['criteria_types']
    expert_weights = st.session_state.trust_data['expert_weights']

    decision_matrix = np.zeros((n_alternatives, n_criteria))
    expert_assessments_matrix = [[[] for _ in range(n_criteria)] for _ in range(n_alternatives)]
    aggregated_tfn_matrix = [[() for _ in range(n_criteria)] for _ in range(n_alternatives)]

    for j, criterion in enumerate(criteria):
        if str(criteria_types[j]).lower() == 'soft':
            for i in range(n_alternatives):
                tfn_list = []
                labels = []
                for e in range(n_experts):
                    expert_df = st.session_state.trust_data['expert_data'][f'expert_{e}']
                    linguistic_value = expert_df.loc[alternatives[i], criterion]
                    labels.append(linguistic_value)
                    tfn_list.append(LINGUISTIC_SCALE[linguistic_value])
                aggregated_tfn = aggregate_tfn(tfn_list, expert_weights)
                decision_matrix[i, j] = defuzz_tfn(aggregated_tfn)
                expert_assessments_matrix[i][j] = labels
                aggregated_tfn_matrix[i][j] = aggregated_tfn
        else:
            for i in range(n_alternatives):
                crisp_value = st.session_state.trust_data['hard_data'].loc[alternatives[i], criterion]
                tfn_value = crisp_to_tfn(crisp_value)
                decision_matrix[i, j] = defuzz_tfn(tfn_value)
                expert_assessments_matrix[i][j] = [crisp_value]
                aggregated_tfn_matrix[i][j] = tfn_value

    st.subheader('Aggregated Decision Matrix Details')
    alt_tabs = st.tabs(alternatives)
    for tab_idx, (tab, alternative) in enumerate(zip(alt_tabs, alternatives)):
        with tab:
            table_data = []
            for j, criterion in enumerate(criteria):
                assessments = expert_assessments_matrix[tab_idx][j]
                aggregated_tfn = aggregated_tfn_matrix[tab_idx][j]
                defuzzified_value = decision_matrix[tab_idx, j]
                if str(criteria_types[j]).lower() == 'soft':
                    assess_str = ', '.join(assessments)
                else:
                    assess_str = f'Crisp: {assessments[0]}'
                table_data.append({
                    'Criterion': criterion,
                    'Type': criteria_types[j],
                    'Expert Assessments': assess_str,
                    'Aggregated TFN': format_tfn(aggregated_tfn),
                    'Defuzzified Value': f'{defuzzified_value:.4f}',
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.subheader('Final Decision Matrix (Defuzzified Values)')
    df_decision = pd.DataFrame(decision_matrix, columns=criteria, index=alternatives)
    st.dataframe(df_decision, use_container_width=True)

    if st.button('Next: Criteria Information'):
        st.session_state.trust_data['decision_matrix'] = decision_matrix
        st.session_state.trust_step = 6
        st.rerun()


def trust_step6_criteria_info():
    st.header('Step 6: Criteria Information')
    n_criteria = st.session_state.trust_data['n_criteria']
    criteria = st.session_state.trust_data['criteria']
    data = {
        'Criterion': criteria,
        'Type': ['Benefit'] * n_criteria,
        'Weight': [1.0 / n_criteria] * n_criteria,
    }
    df = pd.DataFrame(data)
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        column_config={
            'Type': st.column_config.SelectboxColumn('Type', options=['Benefit', 'Cost'])
        }
    )

    weight_sum = edited_df['Weight'].sum()
    if abs(weight_sum - 1.0) > 1e-6:
        st.warning(f'Weights sum to {weight_sum:.4f}, but should sum to 1.0')

    if st.button('Next: Constraint Values'):
        st.session_state.trust_data['criteria_info'] = edited_df
        st.session_state.trust_step = 7
        st.rerun()


def trust_step7_constraints():
    st.header('Step 7: Constraint Values')
    criteria = st.session_state.trust_data['criteria']
    decision_matrix = st.session_state.trust_data['decision_matrix']

    min_vals = np.min(decision_matrix, axis=0)
    max_vals = np.max(decision_matrix, axis=0)
    df = pd.DataFrame({
        'Criterion': criteria,
        'Min Value': min_vals,
        'Max Value': max_vals,
        'ρL': min_vals,
        'ρU': max_vals,
    })
    edited_df = st.data_editor(df, use_container_width=True)

    if st.button('Calculate Results'):
        st.session_state.trust_data['constraints'] = edited_df
        st.session_state.trust_step = 8
        st.rerun()


def trust_step8_calculations():
    st.header('Step 8: TRUST Method Results')
    alternatives = st.session_state.trust_data['alternatives']
    criteria = st.session_state.trust_data['criteria']
    decision_matrix = st.session_state.trust_data['decision_matrix']
    criteria_info = st.session_state.trust_data['criteria_info']
    constraints = st.session_state.trust_data['constraints']
    alpha = st.session_state.trust_data['alpha']
    beta = st.session_state.trust_data['beta']

    n_alternatives = len(alternatives)
    n_criteria = len(criteria)
    criteria_types = criteria_info['Type'].values
    weights = criteria_info['Weight'].values
    LB = constraints['ρL'].values
    UB = constraints['ρU'].values

    all_data = {
        'n_alternatives': n_alternatives,
        'n_criteria': n_criteria,
        'n_experts': st.session_state.trust_data['n_experts'],
        'alpha': alpha,
        'beta': beta,
        'alternatives': alternatives,
        'criteria': criteria,
        'decision_matrix': decision_matrix,
    }

    st.subheader('Step 2.3: Normalization')
    min_vals = np.min(decision_matrix, axis=0)
    max_vals = np.max(decision_matrix, axis=0)

    r_matrix = np.zeros((n_alternatives, n_criteria))
    for j in range(n_criteria):
        denom = max_vals[j] if criteria_types[j] == 'Benefit' else None
        if criteria_types[j] == 'Benefit':
            r_matrix[:, j] = decision_matrix[:, j] / (max_vals[j] if max_vals[j] != 0 else 1)
        else:
            min_val = min_vals[j] if min_vals[j] != 0 else 1e-9
            safe_col = np.where(decision_matrix[:, j] == 0, 1e-9, decision_matrix[:, j])
            r_matrix[:, j] = min_val / safe_col
    st.dataframe(pd.DataFrame(r_matrix, columns=criteria, index=alternatives), use_container_width=True)
    all_data['r_matrix'] = r_matrix

    s_matrix = np.zeros((n_alternatives, n_criteria))
    for j in range(n_criteria):
        if criteria_types[j] == 'Benefit':
            sum_val = np.sum(decision_matrix[:, j])
            s_matrix[:, j] = decision_matrix[:, j] / (sum_val if sum_val != 0 else 1)
        else:
            safe_col = np.where(decision_matrix[:, j] == 0, 1e-9, decision_matrix[:, j])
            sum_recip = np.sum(1 / safe_col)
            s_matrix[:, j] = (1 / safe_col) / (sum_recip if sum_recip != 0 else 1)
    st.dataframe(pd.DataFrame(s_matrix, columns=criteria, index=alternatives), use_container_width=True)
    all_data['s_matrix'] = s_matrix

    m_matrix = np.zeros((n_alternatives, n_criteria))
    for j in range(n_criteria):
        denom = (max_vals[j] - min_vals[j]) if (max_vals[j] - min_vals[j]) != 0 else 1
        if criteria_types[j] == 'Benefit':
            m_matrix[:, j] = (decision_matrix[:, j] - min_vals[j]) / denom
        else:
            m_matrix[:, j] = (max_vals[j] - decision_matrix[:, j]) / denom
    st.dataframe(pd.DataFrame(m_matrix, columns=criteria, index=alternatives), use_container_width=True)
    all_data['m_matrix'] = m_matrix

    l_matrix = np.zeros((n_alternatives, n_criteria))
    for j in range(n_criteria):
        safe_col = np.where(decision_matrix[:, j] <= 0, 1e-9, decision_matrix[:, j])
        product = np.prod(safe_col)
        denom = np.log(product) if product > 0 and np.log(product) != 0 else 1
        for i in range(n_alternatives):
            l_matrix[i, j] = np.log(safe_col[i]) / denom
    st.dataframe(pd.DataFrame(l_matrix, columns=criteria, index=alternatives), use_container_width=True)
    all_data['l_matrix'] = l_matrix

    h_matrix = alpha[0] * r_matrix + alpha[1] * s_matrix + alpha[2] * m_matrix + alpha[3] * l_matrix
    st.write('Aggregated Normalized Matrix (h_ij)')
    st.dataframe(pd.DataFrame(h_matrix, columns=criteria, index=alternatives), use_container_width=True)
    all_data['h_matrix'] = h_matrix

    st.subheader('Step 2.4: Constraint-based Normalization')
    f_matrix = np.zeros((n_alternatives, n_criteria))
    for j in range(n_criteria):
        for i in range(n_alternatives):
            d_ij = decision_matrix[i, j]
            lb_j = LB[j]
            ub_j = UB[j]
            min_j = min_vals[j]
            max_j = max_vals[j]
            max_denom = max(lb_j - min_j, max_j - ub_j, 1e-9)

            if criteria_types[j] == 'Benefit':
                if lb_j <= d_ij <= ub_j:
                    f_matrix[i, j] = 1.0
                elif d_ij < lb_j:
                    f_matrix[i, j] = 1 - (lb_j - d_ij) / (max_denom + 1)
                else:
                    f_matrix[i, j] = 1 - (d_ij - ub_j) / (max_denom + 1)
            else:
                if lb_j <= d_ij <= ub_j:
                    f_matrix[i, j] = 1.0
                elif d_ij < lb_j:
                    f_matrix[i, j] = max(0, 1 - (lb_j - d_ij) / (max_denom + 1))
                else:
                    f_matrix[i, j] = max(0, 1 - (d_ij - ub_j) / (max_denom + 1))
    st.dataframe(pd.DataFrame(f_matrix, columns=criteria, index=alternatives), use_container_width=True)
    all_data['f_matrix'] = f_matrix

    eta_matrix = h_matrix * f_matrix
    st.subheader('Step 2.5: Constrained Aggregated Score Matrix (η_ij)')
    st.dataframe(pd.DataFrame(eta_matrix, columns=criteria, index=alternatives), use_container_width=True)
    all_data['eta_matrix'] = eta_matrix

    v_matrix = eta_matrix * weights
    st.subheader('Step 2.6: Weighted Decision Matrix (v_ij)')
    st.dataframe(pd.DataFrame(v_matrix, columns=criteria, index=alternatives), use_container_width=True)
    all_data['v_matrix'] = v_matrix

    tau = np.min(v_matrix, axis=0)
    st.subheader('Step 2.7: Negative-Ideal Solution (τ_j)')
    st.dataframe(pd.DataFrame([tau], columns=criteria, index=['τ']), use_container_width=True)
    all_data['tau'] = tau

    st.subheader('Step 2.8: Distance Measures')
    epsilon = np.sqrt(np.sum((v_matrix - tau) ** 2, axis=1))
    pi_vals = np.sum(np.abs(v_matrix - tau), axis=1)
    l_distance = np.sum(np.log10(1 + np.abs(v_matrix - tau)), axis=1)
    tau_safe = np.where(tau == 0, 1e-9, tau)
    rho = np.sum(((v_matrix - tau) ** 2) / tau_safe, axis=1)

    distances = pd.DataFrame({
        'Euclidean (ε)': epsilon,
        'Manhattan (π)': pi_vals,
        'Lorentzian (l)': l_distance,
        'Pearson (ρ)': rho,
    }, index=alternatives)
    st.dataframe(distances, use_container_width=True)
    all_data['distances'] = distances

    st.subheader('Step 2.9: Relative Assessment Matrices')
    wp_sum = np.zeros(n_alternatives)
    H_sum = np.zeros(n_alternatives)
    for i in range(n_alternatives):
        for k in range(n_alternatives):
            wp_ik = (epsilon[i] - epsilon[k]) + ((epsilon[i] - epsilon[k]) * (pi_vals[i] - pi_vals[k]))
            H_ik = (l_distance[i] - l_distance[k]) + ((l_distance[i] - l_distance[k]) * (rho[i] - rho[k]))
            wp_sum[i] += wp_ik
            H_sum[i] += H_ik

    rel_assessment = pd.DataFrame({
        'Alternative': alternatives,
        '∑Pik': wp_sum,
        '∑Hik': H_sum,
    })
    st.dataframe(rel_assessment, use_container_width=True)
    all_data['rel_assessment'] = rel_assessment

    st.subheader('Step 2.10: Final Scores and Ranking')
    L_score = beta * wp_sum + (1 - beta) * H_sum
    results = pd.DataFrame({
        'Alternative': alternatives,
        '∑Pik': wp_sum,
        '∑Hik': H_sum,
        'L Score': L_score,
    }).sort_values('L Score', ascending=False)
    results['Rank'] = range(1, len(results) + 1)
    st.dataframe(results, use_container_width=True)

    best_alt = results.iloc[0]['Alternative']
    best_score = results.iloc[0]['L Score']
    st.success(f'Best Alternative: {best_alt} with score: {best_score:.4f}')

    all_data['final_results'] = results
    all_data['best_alternative'] = best_alt
    all_data['best_score'] = best_score

    doc_bytes = create_trust_word_document(all_data)
    st.download_button(
        label='Export TRUST Results to Word',
        data=doc_bytes,
        file_name='Triangular_Fuzzy_TRUST_Results.docx',
        mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        use_container_width=True,
    )

    if st.button('Start Over'):
        st.session_state.trust_step = 1
        st.session_state.trust_data = {}
        st.rerun()


# =========================================================
# MAIN
# =========================================================
def main():
    st.sidebar.title('Fuzzy MCDM Model Selection')
    st.sidebar.markdown('Select the model you want to use:')

    model_choice = st.sidebar.radio(
        'Choose Model:',
        ['Triangular Fuzzy OPA', 'Triangular Fuzzy TRUST Method'],
        index=0,
    )

    if model_choice == 'Triangular Fuzzy OPA':
        opa_model()
    else:
        trust_model()

    st.markdown(
        """
        <div class="footer">
        <p>Integrated Fuzzy MCDM Models | Updated to TFN scale for Moktadir, M. A.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
