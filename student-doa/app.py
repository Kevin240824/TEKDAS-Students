"""Streamlit teaching dashboard: Data -> BI -> ML -> local LLM."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_prep import (
    CLASS_ORDER,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_dataset,
)
from ml_model import train_model
from vai_analyst import DEFAULT_MODEL, build_prompt, call_ollama


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Amazon Rating: BI + ML + LLM", layout="wide")
st.title("Amazon Product Reviews — Data → BI → ML → LLM")
st.caption(
    "Dashboard pembelajaran berbasis dataset Amazon: jelajahi kualitas data, "
    "ringkas pola rating, prediksi kelas rating, lalu interpretasikan bukti "
    "dengan Ollama lokal."
)


@st.cache_data
def load_dataset():
    return prepare_dataset()


@st.cache_resource
def load_training_results(dataset: pd.DataFrame):
    return train_model(dataset)


try:
    products, quality_report = load_dataset()
    training = load_training_results(products)
except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Gagal menyiapkan dataset/model: {error}")
    st.stop()

pipeline = training["pipeline"]
metrics = training["metrics"]
feature_importance = training["feature_importance"]
test_predictions = training["predictions"]
train_ids = set(training["train"]["product_id"].dropna())
test_ids = set(training["test"]["product_id"].dropna())

sidebar = st.sidebar
sidebar.header("Pilih produk")
product_options = products["product_id"].drop_duplicates().tolist()
selected_product_id = sidebar.selectbox(
    "Product ID",
    product_options,
    format_func=lambda product_id: (
        f"{product_id} — "
        f"{products.loc[products['product_id'].eq(product_id), 'product_name'].iloc[0]}"
    ),
)
selected_rows = products.loc[
    products["product_id"].eq(selected_product_id)
]
product = selected_rows.iloc[0]
product_features = selected_rows.iloc[[0]][FEATURE_COLUMNS]
product_probabilities = pipeline.predict_proba(product_features)[0]
model_classes = pipeline.named_steps["classifier"].classes_.tolist()
probability_by_class = {
    label: float(probability)
    for label, probability in zip(model_classes, product_probabilities)
}
probability_by_class = {
    label: probability_by_class[label]
    for label in CLASS_ORDER
}
predicted_class = max(probability_by_class, key=probability_by_class.get)
product_status = (
    "Train"
    if selected_product_id in train_ids
    else "Test"
    if selected_product_id in test_ids
    else "Tidak diketahui"
)

tab_data, tab_bi, tab_ml, tab_llm = st.tabs(
    ["1 · Data", "2 · Business Intelligence", "3 · Machine Learning", "4 · LLM Analyst"]
)

with tab_data:
    st.subheader("Dataset dan kualitas data")
    st.write(
        f"Dataset bersih memiliki **{len(products):,} baris × "
        f"{products.shape[1]} kolom**."
    )
    dropped_rating_rows = quality_report["rows_dropped_invalid_rating"]
    data_col1, data_col2, data_col3 = st.columns(3)
    data_col1.metric("Baris rating rusak dibuang", f"{dropped_rating_rows:,}")
    data_col2.metric(
        "Baris duplikat penuh",
        f"{quality_report['duplicate_rows']:,}",
    )
    data_col3.metric(
        "Nilai kosong tersisa",
        f"{int(products.isna().sum().sum()):,}",
    )
    st.markdown("**Nilai kosong per kolom setelah pembersihan**")
    missing = (
        products.isna()
        .sum()
        .rename("Nilai kosong")
        .to_frame()
        .sort_values("Nilai kosong", ascending=False)
    )
    st.dataframe(missing, use_container_width=True)
    st.markdown("**Kualitas konversi numerik**")
    numeric_quality = pd.DataFrame(
        {
            "Kolom": [
                "discounted_price",
                "actual_price",
                "discount_percentage",
                "rating_count",
                "rating",
            ],
            "Tipe hasil": [
                str(products[column].dtype)
                for column in [
                    "discounted_price",
                    "actual_price",
                    "discount_percentage",
                    "rating_count",
                    "rating",
                ]
            ],
            "Nilai kosong": [
                int(products[column].isna().sum())
                for column in [
                    "discounted_price",
                    "actual_price",
                    "discount_percentage",
                    "rating_count",
                    "rating",
                ]
            ],
        }
    )
    st.dataframe(numeric_quality, use_container_width=True, hide_index=True)
    st.markdown("**Isi dataset bersih**")
    st.dataframe(products, use_container_width=True, height=500)

with tab_bi:
    st.subheader("Business Intelligence deskriptif")
    left, right = st.columns(2)
    with left:
        st.markdown("**Distribusi Rating_Class**")
        class_counts = (
            products[TARGET_COLUMN]
            .value_counts()
            .reindex(CLASS_ORDER, fill_value=0)
        )
        st.bar_chart(class_counts)
    with right:
        st.markdown("**Rata-rata diskon per kategori utama**")
        discount_by_category = (
            products.groupby("main_category")["discount_percentage"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(discount_by_category)

    st.markdown("**Hubungan harga asli dan rating**")
    st.scatter_chart(
        products.dropna(subset=["actual_price"]),
        x="actual_price",
        y="rating",
        color="Rating_Class",
    )

    st.markdown("**Produk dengan rating_count terbanyak**")
    most_rated = (
        products.sort_values("rating_count", ascending=False)
        .drop_duplicates("product_id")
        .head(10)
    )
    st.dataframe(
        most_rated[
            [
                "product_id",
                "product_name",
                "main_category",
                "rating",
                "rating_count",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab_ml:
    st.subheader("Klasifikasi kelas rating produk")
    st.caption(
        f"Produk terpilih berstatus **{product_status}**. "
        "Seluruh baris dengan product_id yang sama berada di partisi yang sama."
    )
    st.markdown(f"**{product['product_name']}** · {product['main_category']}")
    ml_col1, ml_col2 = st.columns([1, 1])
    with ml_col1:
        st.markdown("**Probabilitas per kelas**")
        probability_table = pd.DataFrame(
            {
                "Kelas": CLASS_ORDER,
                "Probabilitas": [
                    probability_by_class[label] for label in CLASS_ORDER
                ],
            }
        )
        st.dataframe(
            probability_table.style.format({"Probabilitas": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.metric("Prediksi model", predicted_class)
        st.metric("Label aktual", str(product[TARGET_COLUMN]))
    with ml_col2:
        st.markdown("**Evaluasi pada test set**")
        metric1, metric2 = st.columns(2)
        metric1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        metric2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        metric3, metric4 = st.columns(2)
        metric3.metric("Dummy accuracy", f"{metrics['dummy_accuracy']:.3f}")
        metric4.metric("Dummy macro F1", f"{metrics['dummy_macro_f1']:.3f}")
        if metrics["accuracy"] < metrics["dummy_accuracy"]:
            st.warning(
                "Accuracy Random Forest masih di bawah baseline DummyClassifier. "
                "Bandingkan juga macro F1 dan metrik tiap kelas sebelum "
                "menyimpulkan model lebih baik."
            )
        st.caption(
            f"Split aktual {metrics['train_rows']:,} train / "
            f"{metrics['test_rows']:,} test baris "
            f"({metrics['train_ratio']:.1%} / {metrics['test_ratio']:.1%}). "
            "StratifiedGroupKFold menjaga pemisahan product_id; ukuran tepat "
            "80/20 dapat sedikit berubah karena grup produk."
        )

    st.markdown("**Confusion matrix — label aktual × prediksi**")
    confusion = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"Aktual {label}" for label in CLASS_ORDER],
        columns=[f"Prediksi {label}" for label in CLASS_ORDER],
    )
    st.dataframe(confusion, use_container_width=True)

    class_metrics = pd.DataFrame(
        {
            label: {
                "Recall": metrics["class_report"][label]["recall"],
                "F1": metrics["class_report"][label]["f1-score"],
            }
            for label in CLASS_ORDER
        }
    ).T
    st.markdown("**Recall dan F1 per kelas**")
    st.dataframe(
        class_metrics.style.format("{:.3f}"),
        use_container_width=True,
    )

    st.markdown("**Feature importance global**")
    st.bar_chart(
        feature_importance.head(15).set_index("feature")["importance"]
    )
    st.download_button(
        "Unduh prediksi test set (CSV)",
        data=test_predictions.to_csv(index=False).encode("utf-8"),
        file_name="amazon_test_predictions.csv",
        mime="text/csv",
    )

with tab_llm:
    st.subheader("LLM Analyst — Ollama lokal")
    st.caption(
        f"Model: {DEFAULT_MODEL}. Ollama opsional; instal dan jalankan "
        "`ollama serve`, lalu unduh model, misalnya `ollama pull llama3.2`."
    )
    model_factors = feature_importance.head(8).to_dict("records")
    prompt = build_prompt(
        product=product.to_dict(),
        probabilities=probability_by_class,
        model_factors=model_factors,
    )
    with st.expander("Lihat prompt yang akan dikirim"):
        st.code(prompt, language="text")
    if st.button("Analisis produk dengan Ollama"):
        with st.spinner("Mengirim data produk ke Ollama lokal..."):
            try:
                answer = call_ollama(prompt)
            except RuntimeError as error:
                st.error(
                    f"Analisis LLM gagal: {error}. Periksa `ollama serve` dan "
                    f"model `{DEFAULT_MODEL}`."
                )
            else:
                st.markdown(answer)

st.markdown("---")
st.caption(
    "Rating_Class adalah label turunan dari rating; rating dan Rating_Class "
    "dikeluarkan dari fitur model untuk mencegah kebocoran target."
)
