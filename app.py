import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# -----------------------------------------------------------------------
# Konfigurasi halaman
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentasi Rasio Harga vs Performa Mobil",
    page_icon="🚗",
    layout="wide",
)

SEGMEN_COLORS = {
    "High Value": "#2ca02c",
    "Standard Value": "#1f77b4",
    "Overpriced / Low Value": "#d62728",
}
SEGMEN_ORDER = ["High Value", "Standard Value", "Overpriced / Low Value"]


# -----------------------------------------------------------------------
# Load artifact (cached supaya tidak reload tiap interaksi)
# -----------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("model_segmentasi_mobil.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv("hasil_segmentasi_mobil.csv")
    return df


bundle = load_bundle()
df = load_data()

reg = bundle["reg"]
scaler = bundle["scaler"]
kmeans = bundle["kmeans"]
label_map = bundle["label_map"]


def prediksi_segmen(hp: float, mpg: float, msrp: float):
    """Sama persis dengan fungsi prediksi_segmen di notebook (bagian Deployment)."""
    if hp <= 0 or mpg <= 0 or msrp <= 0:
        raise ValueError("hp, mpg, dan msrp harus > 0")

    warning = None
    if msrp > 300_000:
        warning = "Harga di atas $300k berada di luar cakupan data latih."

    pred_log_msrp = reg.predict([[np.log(hp), mpg]])[0]
    value_gap = np.log1p(msrp) - pred_log_msrp

    x = pd.DataFrame(
        [[hp, mpg, value_gap]], columns=["Engine HP", "highway MPG", "value_gap"]
    )
    x_scaled = scaler.transform(x)
    cluster = kmeans.predict(x_scaled)[0]
    segmen = label_map[cluster]
    return segmen, round(float(value_gap), 3), warning

# -----------------------------------------------------------------------
# Sidebar Navigasi (Versi Estetik dengan Custom CSS)
# -----------------------------------------------------------------------
# Styling CSS khusus untuk merubah radio button menjadi kartu navigasi
st.sidebar.markdown(
    """
    <style>
    /* Styling kontainer radio button */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 10px;
    }
    
    /* Mengubah item radio menjadi tombol/kartu */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.25s ease-in-out;
        cursor: pointer;
        width: 100%;
    }
    
    /* Efek hover saat kursor diarahkan ke menu */
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: rgba(255, 255, 255, 0.12);
        border-color: #1f77b4;
        transform: translateX(4px);
    }
    
    /* Sembunyikan icon lingkaran radio bawaan Streamlit */
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    /* Teks dalam menu */
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-weight: 500;
        font-size: 15px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Sidebar
st.sidebar.markdown("### 🚗 **Mobil Analytics**")
st.sidebar.caption("Segmentasi Rasio Harga vs Performa")
st.sidebar.markdown("---")

# Menu Navigasi
page = st.sidebar.radio(
    "Halaman",
    ["📊 Explorer Data", "🔮 Prediksi Segmen"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# Informasi Model (Dikemas dalam Expander/Card agar rapi)
with st.sidebar.expander("ℹ️ **Detail Model**", expanded=False):
    st.caption(
        "**Model:** K-Means ($K=3$)\n\n"
        "**Fitur:** Engine HP, highway MPG, dan `value_gap` "
        "(residual regresi $\\log(\\text{MSRP})$ terhadap $\\log(\\text{HP})$ dan MPG).\n\n"
        "**Cakupan Data:** 507 model mobil (2010–2017), non-listrik, MSRP ≤ $300.000."
    )

# =========================================================================
# HALAMAN 1: EXPLORER DATA
# =========================================================================
if page == "📊 Explorer Data":
    st.title("📊 Explorer Segmentasi Mobil")
    st.caption(
        "Eksplorasi 507 model mobil yang sudah dikelompokkan ke tiga segmen "
        "berdasarkan rasio harga terhadap performa (Engine HP & highway MPG)."
    )

    # --- Filter ---
    with st.expander("🔍 Filter", expanded=True):
        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            makes = sorted(df["Make"].unique())
            pilih_make = st.multiselect("Merek (Make)", makes, default=[])
        with c2:
            pilih_segmen = st.multiselect(
                "Segmen", SEGMEN_ORDER, default=SEGMEN_ORDER
            )
        with c3:
            hp_min, hp_max = int(df["Engine HP"].min()), int(df["Engine HP"].max())
            rentang_hp = st.slider("Engine HP", hp_min, hp_max, (hp_min, hp_max))

        c4, c5 = st.columns(2)
        with c4:
            msrp_min, msrp_max = int(df["MSRP"].min()), int(df["MSRP"].max())
            rentang_msrp = st.slider(
                "MSRP ($)", msrp_min, msrp_max, (msrp_min, msrp_max), step=1000
            )
        with c5:
            mpg_min, mpg_max = int(df["highway MPG"].min()), int(
                df["highway MPG"].max()
            )
            rentang_mpg = st.slider("Highway MPG", mpg_min, mpg_max, (mpg_min, mpg_max))

    df_filt = df[
        df["Segmen"].isin(pilih_segmen)
        & df["Engine HP"].between(*rentang_hp)
        & df["MSRP"].between(*rentang_msrp)
        & df["highway MPG"].between(*rentang_mpg)
    ]
    if pilih_make:
        df_filt = df_filt[df_filt["Make"].isin(pilih_make)]

    st.markdown(f"**{len(df_filt)}** dari {len(df)} model ditampilkan.")

    # --- Ringkasan metrik per segmen ---
    st.subheader("Ringkasan per Segmen")
    ringkasan = (
        df_filt.groupby("Segmen")
        .agg(
            Jumlah=("Model", "count"),
            HP_median=("Engine HP", "median"),
            MPG_median=("highway MPG", "median"),
            MSRP_median=("MSRP", "median"),
            value_gap_median=("value_gap", "median"),
        )
        .reindex(SEGMEN_ORDER)
        .dropna(how="all")
        .round(2)
    )
    cols = st.columns(len(ringkasan)) if len(ringkasan) > 0 else []
    for col, (seg, row) in zip(cols, ringkasan.iterrows()):
        with col:
            st.markdown(f"**{seg}**")
            st.metric("Jumlah model", int(row["Jumlah"]))
            st.metric("Median MSRP", f"${row['MSRP_median']:,.0f}")
            st.metric("Median HP / MPG", f"{row['HP_median']:.0f} / {row['MPG_median']:.0f}")

    # --- Scatter plot ---
    st.subheader("Sebaran Engine HP vs MSRP")
    fig = px.scatter(
        df_filt,
        x="Engine HP",
        y="MSRP",
        color="Segmen",
        color_discrete_map=SEGMEN_COLORS,
        category_orders={"Segmen": SEGMEN_ORDER},
        hover_data=["Make", "Model", "highway MPG", "value_gap"],
        log_y=True,
        labels={"MSRP": "MSRP ($, skala log)"},
    )
    fig.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=0.5, color="white")))
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    # --- Distribusi value_gap ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribusi value_gap per Segmen")
        fig2 = px.box(
            df_filt,
            x="Segmen",
            y="value_gap",
            color="Segmen",
            color_discrete_map=SEGMEN_COLORS,
            category_orders={"Segmen": SEGMEN_ORDER},
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.subheader("Jumlah Model per Segmen")
        jumlah = df_filt["Segmen"].value_counts().reindex(SEGMEN_ORDER).fillna(0)
        fig3 = px.bar(
            jumlah,
            x=jumlah.index,
            y=jumlah.values,
            color=jumlah.index,
            color_discrete_map=SEGMEN_COLORS,
            labels={"x": "Segmen", "y": "Jumlah model"},
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # --- Tabel data ---
    st.subheader("Tabel Data")
    kolom_tampil = [
        "Make", "Model", "Engine HP", "highway MPG", "MSRP", "value_gap", "Segmen",
    ]
    st.dataframe(
        df_filt[kolom_tampil].sort_values("value_gap"),
        use_container_width=True,
        hide_index=True,
    )

# =========================================================================
# HALAMAN 2: PREDIKSI SEGMEN
# =========================================================================
else:
    st.title("🔮 Prediksi Segmen Mobil Baru")
    st.caption(
        "Masukkan spesifikasi mobil untuk memprediksi segmen rasio harga-performanya "
        "menggunakan model K-Means yang sudah dilatih."
    )

    with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        with c1:
            hp = st.number_input("Engine HP", min_value=1.0, value=150.0, step=5.0)
        with c2:
            mpg = st.number_input("Highway MPG", min_value=1.0, value=35.0, step=1.0)
        with c3:
            msrp = st.number_input("MSRP ($)", min_value=1.0, value=22000.0, step=500.0)
        submitted = st.form_submit_button("Prediksi Segmen", use_container_width=True)

    if submitted:
        try:
            segmen, value_gap, warning = prediksi_segmen(hp, mpg, msrp)
        except ValueError as e:
            st.error(str(e))
        else:
            if warning:
                st.warning(warning)

            warna = SEGMEN_COLORS.get(segmen, "#888888")
            st.markdown(
                f"""
                <div style="padding:1.2rem;border-radius:0.6rem;background-color:{warna}22;
                            border:1.5px solid {warna};">
                    <h3 style="margin:0;color:{warna};">Segmen: {segmen}</h3>
                    <p style="margin:0.3rem 0 0 0;">value_gap = <b>{value_gap}</b>
                    (residual log-harga terhadap performa &mdash; positif berarti lebih mahal
                    dari harga lazim, negatif berarti lebih murah)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Posisi Mobil Baru Relatif terhadap Data Latih")
            df_plot = df.copy()
            df_plot["Tipe"] = "Data Latih"
            baris_baru = pd.DataFrame(
                [{
                    "Engine HP": hp,
                    "MSRP": msrp,
                    "highway MPG": mpg,
                    "value_gap": value_gap,
                    "Segmen": segmen,
                    "Tipe": "Mobil Baru",
                    "Make": "-",
                    "Model": "Input Baru",
                }]
            )
            df_plot = pd.concat([df_plot, baris_baru], ignore_index=True)

            fig = px.scatter(
                df_plot,
                x="Engine HP",
                y="MSRP",
                color="Segmen",
                color_discrete_map=SEGMEN_COLORS,
                category_orders={"Segmen": SEGMEN_ORDER},
                symbol="Tipe",
                symbol_map={"Data Latih": "circle", "Mobil Baru": "star"},
                size=df_plot["Tipe"].map({"Data Latih": 8, "Mobil Baru": 22}),
                hover_data=["Make", "Model", "highway MPG", "value_gap"],
                log_y=True,
                labels={"MSRP": "MSRP ($, skala log)"},
            )
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

    st.info(
        "⚠️ **Keterbatasan model**: dilatih pada data mobil AS tahun 2010-2017 (harga nominal USD saat itu), "
        "hanya mencakup mobil non-listrik dengan MSRP ≤ $300.000. Prediksi untuk mobil dengan harga terkini "
        "atau pasar di luar cakupan tersebut (mis. harga Indonesia) tidak divalidasi dan sebaiknya "
        "diinterpretasikan dengan hati-hati."
    )
