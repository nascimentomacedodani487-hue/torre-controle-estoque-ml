import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Mercado Livre - Control Tower CD SP04", 
    layout="wide", 
    page_icon="💛",
    initial_sidebar_state="expanded"
)

# Estilização CSS no padrão Mercado Livre
st.markdown("""
<style>
    .metric-card {
        background-color: #FFFDE7;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #FFE600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stAppHeader {background-color: #FFE600;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# 1. Carregamento de Dados e Cache
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    df_feat = pd.read_parquet("df_feat.parquet")
    df_raw = pd.read_parquet("df_raw.parquet")
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    return df_feat, df_raw

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

df_feat, df_raw = load_data()
xgb_model = load_model()
br_holidays = holidays.Brazil(years=[2024, 2025, 2026])

FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_30", 
    "roll_mean_7", "roll_std_7", "roll_mean_30", "roll_std_30", 
    "day_of_week", "month", "is_weekend", "is_holiday", "day_of_year"
]

# --------------------------------------------------------------------------------------
# 2. Sidebar — Controles Adaptados ao CD SP04
# --------------------------------------------------------------------------------------
st.sidebar.image("https://http2.mlstatic.com/frontend-assets/ui-navigation/5.22.8/mercadolibre/logo__large_plus.png", width=160)
st.sidebar.title("🟡 Torre de Controle SP04")
st.sidebar.caption("📍 CD Cajamar (SP) — Complexo Logístico Fulfillment")

st.sidebar.subheader("📍 Módulos & Categorias")

hubs_available = sorted(df_raw["hub"].dropna().unique().tolist())
skus_available = sorted(df_raw["sku"].dropna().unique().tolist())

all_hubs = st.sidebar.checkbox("Selecionar Todos os Módulos", value=True, key="cb_all_hubs")
if all_hubs:
    hubs_selected = st.sidebar.multiselect("Módulo do CD", options=hubs_available, default=hubs_available, key="mult_hubs")
else:
    hubs_selected = st.sidebar.multiselect("Módulo do CD", options=hubs_available, default=[hubs_available[0]], key="mult_hubs")

all_skus = st.sidebar.checkbox("Selecionar Todos os SKUs", value=True, key="cb_all_skus")
if all_skus:
    skus_selected = st.sidebar.multiselect("SKU Mercado Livre", options=skus_available, default=skus_available, key="mult_skus")
else:
    skus_selected = st.sidebar.multiselect("SKU Mercado Livre", options=skus_available, default=[skus_available[0]], key="mult_skus")

horizon = st.sidebar.selectbox("Horizonte Preditivo (Dias)", [7, 14, 30], index=2, key="horizon_key")

st.sidebar.divider()
st.sidebar.subheader("🚚 Parâmetros de Suprimentos")
lead_time = st.sidebar.slider("Lead Time do Fornecedor (dias)", 1, 30, 5, key="lt_key")
service_level_label = st.sidebar.select_slider(
    "Nível de Serviço Alvo (SLA)", 
    options=["90%", "95%", "97.5%", "99%"], 
    value="97.5%",
    key="sl_key"
)

st.sidebar.divider()
st.sidebar.subheader("💰 Premissas Financeiras")
unit_cost = st.sidebar.number_input("Custo Médio Unitário (R$)", min_value=1.0, value=45.0, step=5.0, key="cost_key")
holding_cost_pct = st.sidebar.slider("Custo de Posse de Estoque (%/ano)", 5, 40, 18, key="holding_key") / 100
order_fixed_cost = st.sidebar.number_input("Custo Fixo por Pedido (R$)", min_value=10.0, value=150.0, step=10.0, key="order_key")

Z_MAP = {"90%": 1.28, "95%": 1.65, "97.5%": 1.96, "99%": 2.33}
Z = Z_MAP[service_level_label]

if not hubs_selected or not skus_selected:
    st.warning("⚠️ Selecione pelo menos um Módulo e um SKU na barra lateral para carregar o painel.")
    st.stop()

# --------------------------------------------------------------------------------------
# 3. Filtragem e Agregação dos Dados em Tempo Real
# --------------------------------------------------------------------------------------
df_filtered = df_raw[
    (df_raw["hub"].isin(hubs_selected)) & 
    (df_raw["sku"].isin(skus_selected))
].sort_values("date")

if df_filtered.empty:
    st.warning("Nenhum registro encontrado para essa combinação de Módulos e SKUs no CD SP04.")
    st.stop()

serie = (
    df_filtered.groupby("date")["demand_clean"]
    .sum()
    .asfreq("D")
    .ffill()
)

def generate_forecast(history_series, model, n_days):
    hist_vals = list(history_series.values)
    last_date = history_series.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days, freq="D")
    
    preds = []
    for cur_date in future_dates:
        row = [
            hist_vals[-1], hist_vals[-7], hist_vals[-14], hist_vals[-30],
            np.mean(hist_vals[-7:]), np.std(hist_vals[-7:]),
            np.mean(hist_vals[-30:]), np.std(hist_vals[-30:]),
            cur_date.dayofweek, cur_date.month,
            int(cur_date.dayofweek in [5, 6]),
            int(cur_date.date() in br_holidays),
            cur_date.dayofyear
        ]
        X_df = pd.DataFrame([row], columns=FEATURE_COLS)
        p = max(float(model.predict(X_df)[0]), 0.0)
        preds.append(p)
        hist_vals.append(p)
        
    return pd.Series(preds, index=future_dates)

forecast_days = max(int(horizon), int(lead_time))
forecast = generate_forecast(serie, xgb_model, forecast_days)

# --------------------------------------------------------------------------------------
# 4. Interface por Abas
# --------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Painel Geral CD SP04", 
    "🔮 Previsão de Expedição (ML)", 
    "⚙️ Dimensionamento de Estoque", 
    "📑 Plano de Abastecimento & Transferências"
])

# --------------------------------------------------------------------------------------
# ABA 1: PAINEL GERAL CD SP04
# --------------------------------------------------------------------------------------
with tab1:
    st.title("📦 Centro de Distribuição SP04 — Cajamar (Mercado Livre)")
    st.caption(f"Exibindo dados para **{len(hubs_selected)} Módulo(s)** e **{len(skus_selected)} SKU(s)** selecionado(s).")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Módulos Selecionados", f"{len(hubs_selected)} de {len(hubs_available)}")
    c2.metric("SKUs Selecionados", f"{len(skus_selected)} de {len(skus_available)}")
    c3.metric("Expedição Média Diária", f"{serie.mean():,.0f} un./dia")
    c4.metric("Volume Histórico Total", f"{df_filtered['demand_clean'].sum():,.0f} un.")

    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Expedição Diária por Módulo (CD SP04)")
        df_hub_time = df_filtered.groupby(["date", "hub"])["demand_clean"].sum().reset_index()
        df_hub_time["demand_smooth"] = df_hub_time.groupby("hub")["demand_clean"].transform(lambda x: x.rolling(7, min_periods=1).mean())
        
        fig_hub = px.line(
            df_hub_time, x="date", y="demand_smooth", color="hub", 
            labels={"demand_smooth": "Média de Saídas (7d)", "date": "Data", "hub": "Módulo SP04"},
            template="plotly_dark",
            color_discrete_sequence=["#FFE600", "#2D3277", "#00A650", "#FF7733"]
        )
        fig_hub.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_hub, use_container_width=True)

    with col2:
        st.subheader("📊 Giro de Estoque por Categoria")
        df_abc = df_filtered.groupby("sku")["demand_clean"].sum().reset_index().sort_values("demand_clean", ascending=False)
        fig_abc = px.bar(
            df_abc, x="sku", y="demand_clean", color="demand_clean", 
            color_continuous_scale="YlOrBr", template="plotly_white"
        )
        fig_abc.update_layout(height=380, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_abc, use_container_width=True)

# --------------------------------------------------------------------------------------
# ABA 2: PREVISÃO DE EXPEDIÇÃO
# --------------------------------------------------------------------------------------
with tab2:
    st.title("🔮 Previsão de Demanda & Saídas de Pedidos")
    
    lt_demand = forecast.iloc[:lead_time].sum()
    hz_demand = forecast.iloc[:horizon].sum()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Demanda no Lead Time ({lead_time}d)", f"{lt_demand:,.0f} un.")
    m2.metric(f"Previsão ({horizon}d)", f"{hz_demand:,.0f} un.")
    m3.metric("Média Diária Prevista", f"{forecast.iloc[:horizon].mean():,.1f} un.")
    m4.metric("Desvio Padrão Operacional", f"{serie.tail(60).std():,.2f}")

    st.divider()

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=serie.index[-90:], y=serie.values[-90:], name="Expedição Histórica Real", line=dict(color="#2D3277", width=2)))
    fig_pred.add_trace(go.Scatter(x=forecast.index[:horizon], y=forecast.values[:horizon], name=f"Previsão XGBoost ({horizon}d)", line=dict(color="#00A650", dash="dash", width=2.5)))

    lt_end = forecast.index[min(lead_time - 1, len(forecast) - 1)]
    fig_pred.add_vrect(
        x0=forecast.index[0], x1=lt_end,
        fillcolor="rgba(255, 230, 0, 0.25)", layer="below",
        line_width=1, line_color="rgba(255, 230, 0, 0.8)"
    )
    
    y_max = max(float(serie.tail(90).max()), float(forecast.max())) * 1.05
    fig_pred.add_trace(go.Scatter(
        x=[lt_end, lt_end], y=[0, y_max],
        mode="lines+text", name=f"Fim Lead Time ({lead_time}d)",
        line=dict(color="#D32F2F", width=2, dash="dot"),
        text=["", f" Recebimento ({lead_time}d)"], textposition="top right"
    ))

    fig_pred.update_layout(
        title="Curva de Demanda Prevista vs Histórica — CD SP04 Cajamar",
        template="plotly_white", xaxis_title="Data", yaxis_title="Volume de Pacotes", height=420
    )
    st.plotly_chart(fig_pred, use_container_width=True)

# --------------------------------------------------------------------------------------
# ABA 3: DIMENSIONAMENTO DE ESTOQUE
# --------------------------------------------------------------------------------------
with tab3:
    st.title("⚙️ Otimização de Estoque & Parâmetros Fulfillment")
    
    residuals = serie.tail(60) - serie.tail(60).rolling(7).mean()
    err_std = max(float(residuals.dropna().std()), 0.1)
    avg_d = max(float(serie.tail(90).mean()), 0.1)
    ann_d = avg_d * 365

    ss = int(np.ceil(Z * err_std * np.sqrt(lead_time)))
    rop = int(np.ceil((avg_d * lead_time) + ss))
    holding_unit_year = unit_cost * holding_cost_pct
    eoq = int(np.ceil(np.sqrt((2 * ann_d * order_fixed_cost) / holding_unit_year)))
    capital_ss = ss * unit_cost

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Estoque de Segurança (SS)", f"{ss:,.0f} un.")
    f2.metric("Ponto de Pedido (ROP)", f"{rop:,.0f} un.")
    f3.metric("Lote Econômico (EOQ)", f"{eoq:,.0f} un.")
    f4.metric("Capital Imobilizado em SS", f"R$ {capital_ss:,.2f}")

    st.divider()
    st.subheader("📉 Simulação do Ciclo Dinâmico de Estoque (Dente de Serra)")
    
    sim_days = 60
    timeline = []
    inv = rop + (eoq / 2)
    for _ in range(sim_days):
        inv -= avg_d
        if inv <= ss:
            inv += eoq
        timeline.append(inv)

    sim_df = pd.DataFrame({
        "Data": pd.date_range(pd.Timestamp.now(), periods=sim_days, freq="D"),
        "Estoque_Físico": timeline,
        "ROP": rop,
        "SS": ss
    })

    fig_saw = go.Figure()
    fig_saw.add_trace(go.Scatter(x=sim_df["Data"], y=sim_df["Estoque_Físico"], name="Nível Físico no CD", line=dict(color="#00A650", width=2)))
    fig_saw.add_trace(go.Scatter(x=sim_df["Data"], y=sim_df["ROP"], name=f"Ponto de Pedido (ROP = {rop:,.0f} un)", line=dict(color="#D32F2F", dash="dash")))
    fig_saw.add_trace(go.Scatter(x=sim_df["Data"], y=sim_df["SS"], name=f"Estoque de Segurança (SS = {ss:,.0f} un)", line=dict(color="#FFA000", dash="dot")))
    
    fig_saw.update_layout(template="plotly_white", xaxis_title="Data", yaxis_title="Unidades em Estoque", height=380)
    st.plotly_chart(fig_saw, use_container_width=True)

# --------------------------------------------------------------------------------------
# ABA 4: PLANO DE ABASTECIMENTO
# --------------------------------------------------------------------------------------
with tab4:
    st.title("📑 Plano de Abastecimento para Vendedores / Fulfillment")
    
    df_plan = pd.DataFrame([{
        "Centro de Distribuição": "CD SP04 - Cajamar (SP)",
        "Módulos Selecionados": ", ".join(hubs_selected) if len(hubs_selected) <= 3 else f"{len(hubs_selected)} Módulos",
        "SKUs Selecionados": ", ".join(skus_selected) if len(skus_selected) <= 3 else f"{len(skus_selected)} SKUs",
        "Média de Saída Diária": round(avg_d, 1),
        "Lead Time Reposição": f"{lead_time} dias",
        "SLA de Atendimento": service_level_label,
        "Estoque Segurança Recomendado": f"{ss:,.0f} un",
        "Ponto de Disparo (ROP)": f"{rop:,.0f} un",
        "Lote Padrão de Envio (EOQ)": f"{eoq:,.0f} un",
        "Custo Unitário Médio (R$)": f"R$ {unit_cost:.2f}",
        "Capital Imobilizado (SS)": f"R$ {capital_ss:,.2f}"
    }])
    st.dataframe(df_plan, use_container_width=True, hide_index=True)

    st.subheader("📦 Programação Diária de Recebimento no CD")
    df_detail = forecast.iloc[:horizon].rename("Previsão de Saída").round(1).reset_index()
    df_detail.columns = ["Data", "Previsão de Saída (un)"]
    df_detail["Valor Movimentado Estimado (R$)"] = (df_detail["Previsão de Saída (un)"] * unit_cost).apply(lambda x: f"R$ {x:,.2f}")
    
    st.dataframe(df_detail, use_container_width=True, hide_index=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="📥 Exportar Plano de Abastecimento (CSV)",
            data=df_plan.to_csv(index=False).encode("utf-8"),
            file_name="plano_abastecimento_sp04.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_d2:
        st.download_button(
            label="📥 Exportar Previsão Diária (CSV)",
            data=df_detail.to_csv(index=False).encode("utf-8"),
            file_name="previsao_diaria_sp04.csv",
            mime="text/csv",
            use_container_width=True
        )
