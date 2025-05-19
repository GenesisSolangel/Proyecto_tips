import streamlit as st
import pandas as pd
import re
import seaborn as sns
import plotly.express as px
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = sns.load_dataset("tips")

X = df[["total_bill", "size"]]
y = df[["tip"]]

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

model = LinearRegression()
model.fit(X_scaled, y_scaled)

# Guarda el modelo y los escaladores
with open("tip_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("X_scaler.pkl", "wb") as f:
    pickle.dump(X_scaler, f)

with open("y_scaler.pkl", "wb") as f:
    pickle.dump(y_scaler, f)

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

    st.markdown("## Predicción de propina")

    st.markdown("Ingresa los siguientes valores para predecir la propina estimada:")

    # Inputs del usuario
    col1, col2 = st.columns(2)
    with col1:
        total_bill = st.number_input("Total de la cuenta ($)", min_value=0.0, step=1.0)
    with col2:
        size = st.number_input("Tamaño del grupo", min_value=1, step=1)

    # Botón para predecir
    if st.button("Predecir propina"):
        import numpy as np
        import pickle

        # Cargar modelo y escaladores
        with open("tip_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("X_scaler.pkl", "rb") as f:
            X_scaler = pickle.load(f)
        with open("y_scaler.pkl", "rb") as f:
            y_scaler = pickle.load(f)

        # Crear input y escalar
        data = np.array([total_bill, size]).reshape(1, -1)
        data_scaled = X_scaler.transform(data)

        # Predicción
        prediction_scaled = model.predict(data_scaled)
        prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

        # Mostrar resultados
        df_pred = pd.DataFrame(data=data, columns=["total_bill", "size"])
        df_pred["Predicted Tip ($)"] = prediction

        col1, col2 = st.columns(2)
        col1.markdown("**Datos del usuario:**")
        col1.dataframe(df_pred[["total_bill", "size"]])

        col2.markdown("**Propina estimada:**")
        col2.dataframe(df_pred[["Predicted Tip ($)"]])
if __name__ == "__main__":
    main()
