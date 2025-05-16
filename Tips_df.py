import streamlit as st
import pandas as pd
import re
import seaborn as sns
from textblob import TextBlob
import plotly.express as px
from PIL import Image
from modules.the_office_func import *

st.set_page_config(page_title = "Tips",
                   layout     = "centered")

def main():
    st.title("Tips dataframe analysis")

    # Data
    df = sns.load_dataset("tips")

    #Sidebar
    st.sidebar.markdown("### Filters")

    sex = st.sidebar.radio(label="Filter by sex",
                   options=("All", "Male", "Female"),
                   index=0,
                   disabled=False,
                   horizontal=True,
                   )

    smoker = st.sidebar.radio(label="Smoker?",
                           options=("All", "Yes", "No"),
                           index=0,
                           disabled=False,
                           horizontal=True,
                           )

    day = st.sidebar.selectbox(label="Day",
                                  options=df["day"].unique(),
                                  index=0)

    size_min = st.sidebar.slider(label="Min size",
                        min_value=0,
                        max_value=10,
                        value=0,
                        step=1)

    size_max = st.sidebar.slider(label="Max size",
                                 min_value=0,
                                 max_value=10,
                                 value=0,
                                 step=1)

    if sex != "All":
        df = df[df['sex'] == sex]

    if smoker != "All":
        df = df[df['smoker'] == smoker]

    df = df[df["day"] == day]

    df = df[df['size'].between(size_min, size_max)]

    with st.expander(label="DataFrame", expanded=False):
        st.dataframe(df)

    # Agregar un gráfico con df filtrado

    df_sex_counts = df["sex"].value_counts().reset_index()
    df_sex_counts.columns = ["sex", "count"]

    fig_bar = px.bar(
        data_frame=df_sex_counts,
        x="sex",
        y="count",
        title="Beneficiarios por sexo"
    )
    st.plotly_chart(figure_or_data=fig_bar, use_container_width=True)


    fig_hist = px.histogram(data_frame=df["tip"],
                            x="tip",
                            title="Distribución de propinas")
    st.plotly_chart(figure_or_data=fig_hist, use_container_width=True)

    fig_hist2 = px.histogram(data_frame=df["total_bill"],
                            x="total_bill",
                            title="Distribución del total de cuentas")
    st.plotly_chart(figure_or_data=fig_hist2, use_container_width=True)


if __name__ == "__main__":
    main()
