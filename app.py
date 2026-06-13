import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de página y línea gráfica (Azul institucional)
st.set_page_config(page_title="Portafolio Ciencia de Datos - UGB", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size:38px !important; color:#0c2340; font-weight: bold; text-align: center; margin-bottom:20px; }
    .section-title { color:#0c2340; font-weight: bold; border-bottom: 2px solid #0c2340; padding-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- CARGA DE DATOS AUTOMÁTICA CON RUTA ABSOLUTA ---
@st.cache_data
def cargar_datos():
    try:
        ruta_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_csv = os.path.join(ruta_actual, "anime.csv")
        
        df = pd.read_csv(ruta_csv)
        
        # Conversión y limpieza estricta de las columnas reales de tu dataset
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')
        df['Vote'] = pd.to_numeric(df['Vote'], errors='coerce')
        df['Ranked'] = pd.to_numeric(df['Ranked'], errors='coerce')
        df['Popularity'] = pd.to_numeric(df['Popularity'], errors='coerce')
        
        df = df.dropna(subset=['Score', 'Episodes', 'Title'])
        return df
    except Exception as e:
        st.error(f"Error crítico al cargar 'anime.csv': {e}")
        return pd.DataFrame()

df_anime = cargar_datos()

# --- MENÚ LATERAL ---
with st.sidebar:
    st.title("👨‍💻 Portafolio Digital")
    st.markdown("**Estudiante:** Josué David López Dubón")
    st.markdown("**Carrera:** Ingeniería en Sistemas")
    st.markdown("---")
    
    seleccion = option_menu(
        menu_title="Menú Principal",
        options=[
            "1. Inicio", 
            "2. Análisis Exploratorio", 
            "3. Aprendizaje Automático", 
            "4. Sistema de Recomendación", 
            "5. Carga de Archivos", 
            "6. Análisis de Sentimientos"
        ],
        icons=["house", "search", "cpu", "star", "cloud-upload", "emoji-smile"],
        menu_icon="cast", 
        default_index=0,
    )

# ==========================================================
# OPCIÓN 1: INICIO
# ==========================================================
if seleccion == "1. Inicio":
    st.markdown("<h1 class='main-title'>Portafolio Profesional de Ciencia de Datos</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        # Enlace transformado para descarga directa de Google Drive
        st.image("https://thumbnail.ws.126.net/jpg/https://drive.google.com/uc?export=download&id=1aYk9waqeYXagIR7_gq8dmxnPGB9u5-h4", caption="Josué David López Dubón", width=230)
    with col2:
        st.markdown("<h3 class='section-title'>Resumen Profesional</h3>", unsafe_allow_html=True)
        st.write("""
        Estudiante de la carrera de Ingeniería en Sistemas y Redes Informáticas en la Universidad Gerardo Barrios. 
        Formación orientada al desarrollo de soluciones de software, analítica de datos y estadística computacional, 
        aplicando modelos inteligentes para la optimización de procesos de negocio y plataformas tecnológicas.
        """)
        st.markdown("**Institución:** Universidad Gerardo Barrios (UGB)")
        st.markdown("**Asignatura:** Técnica Electiva I - Ciencia de Datos")

    st.markdown("---")
    st.markdown("<h3 class='section-title'>Demo de Data Storytelling</h3>", unsafe_allow_html=True)
    url_video = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # REEMPLAZAR CON TU LINK REAL
    st.video(url_video)

# ==========================================================
# OPCIÓN 2: ANÁLISIS EXPLORATORIO
# ==========================================================
elif seleccion == "2. Análisis Exploratorio":
    st.markdown("<h1 class='main-title'>Análisis Exploratorio de Datos (EDA)</h1>", unsafe_allow_html=True)
    
    if df_anime.empty:
        st.warning("Dataset vacío o no encontrado.")
    else:
        sub_menu = st.tabs(["Descripción del Dataset", "Descripción de Campos", "Navegador Completo", "Graficador", "Hipótesis"])
        
        with sub_menu[0]:
            st.subheader("Descripción General")
            st.write("Análisis estadístico del rendimiento, popularidad y formatos del ecosistema de animación indexado de Kaggle.")
            st.metric("Registros Procesados", df_anime.shape[0])
            st.metric("Variables Disponibles", df_anime.shape[1])
            
        with sub_menu[1]:
            st.subheader("Descripción Dinámica de Campos")
            campo = st.selectbox("Selecciona un campo:", df_anime.columns)
            if pd.api.types.is_numeric_dtype(df_anime[campo]):
                st.write("**Tipo:** Numérico")
                st.write(df_anime[campo].describe())
            else:
                st.write("**Tipo:** Categórico")
                st.write(df_anime[campo].unique()[:20])
                
        with sub_menu[2]:
            st.subheader("Navegador Completo")
            buscar_id = st.checkbox("Buscar registro por índice (Bonus)")
            if buscar_id:
                codigo = st.number_input("Índice:", min_value=0, max_value=len(df_anime)-1, value=0)
                st.write(df_anime.iloc[[codigo]])
            else:
                st.dataframe(df_anime)
                
        with sub_menu[3]:
            st.subheader("Graficador Automático")
            col_grafica = st.selectbox("Columna a graficar:", ["Score", "Episodes", "Status"])
            if col_grafica == "Score":
                fig = px.histogram(df_anime, x="Score", title="Distribución de Calificaciones", color_discrete_sequence=['#0c2340'])
            elif col_grafica == "Episodes":
                fig = px.box(df_anime, y="Episodes", title="Dispersión de Episodios")
            else:
                fig = px.histogram(df_anime, x="Status", title="Distribución por Estado de Emisión", color="Status")
            st.plotly_chart(fig, use_container_width=True)
                
        with sub_menu[4]:
            st.subheader("Validación de Hipótesis")
            hipotesis = st.radio("Selecciona una hipótesis:", [
                "Hipótesis 1: Las obras con mayor cantidad de votos (Vote) poseen mejores calificaciones (Score).",
                "Hipótesis 2: El orden de popularidad (Popularity) tiene una relación inversa con el Score."
            ])
            if "Hipótesis 1" in hipotesis:
                corr = df_anime['Vote'].corr(df_anime['Score'])
                fig = px.scatter(df_anime, x="Vote", y="Score", title="Relación Votos vs Puntuación")
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"**Correlación:** {corr:.4f}. Se confirma que las comunidades masivas impactan positivamente la nota.")
            else:
                corr = df_anime['Popularity'].corr(df_anime['Score'])
                fig = px.scatter(df_anime, x="Popularity", y="Score", title="Popularidad vs Score")
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"**Correlación:** {corr:.4f}. Al ser un ranking (donde 1 es el más popular), la correlación negativa confirma que a menor número de puesto en popularidad, mayor es el Score.")

# ==========================================================
# OPCIÓN 3: APRENDIZAJE AUTOMÁTICO
# ==========================================================
elif seleccion == "3. Aprendizaje Automático":
    st.markdown("<h1 class='main-title'>Modelos Predictivos (Machine Learning)</h1>", unsafe_allow_html=True)
    
    if df_anime.empty:
        st.warning("Datos no cargados.")
    else:
        algoritmo = st.selectbox("Algoritmo:", ["Regresión Lineal Múltiple", "Árbol de Decisión Regresor"])
        variables_indep = st.multiselect("Variables Predictoras (X):", ["Episodes", "Vote", "Ranked", "Popularity"], default=["Episodes", "Vote"])
        test_size_per = st.slider("Porcentaje de Test (%):", 10, 50, 20, 5)
        
        if len(variables_indep) >= 1:
            X = df_anime[variables_indep]
            y = df_anime["Score"]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_size_per/100), random_state=42)
            
            model = LinearRegression() if algoritmo == "Regresión Lineal Múltiple" else DecisionTreeRegressor(max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            c1, c2 = st.columns(2)
            c1.metric("Coeficiente de Determinación ($R^2$)", f"{r2_score(y_test, preds):.4f}")
            c2.metric("Error Cuadrático (RMSE)", f"{np.sqrt(mean_squared_error(y_test, preds)):.4f}")
            
            df_res = pd.DataFrame({'Reales': y_test, 'Predicciones': preds})
            fig_ml = px.scatter(df_res, x='Reales', y='Predicciones', title="Predicciones vs Valores Reales")
            st.plotly_chart(fig_ml, use_container_width=True)
        else:
            st.error("Selecciona al menos una variable X.")

# ==========================================================
# OPCIÓN 4: SISTEMA DE RECOMENDACIÓN
# ==========================================================
elif seleccion == "4. Sistema de Recomendación":
    st.markdown("<h1 class='main-title'>Sistema de Recomendación Basado en Formato y Origen</h1>", unsafe_allow_html=True)
    
    if df_anime.empty:
        st.warning("Datos no disponibles.")
    else:
        st.write("Selecciona una obra y el sistema te recomendará otras que compartan la misma fuente de origen (Source) y formato (Status).")
        anime_sel = st.selectbox("Selecciona un título:", df_anime['Title'].unique())
        
        # Filtro inteligente adaptado a las columnas reales de tu dataset
        fila_ref = df_anime[df_anime['Title'] == anime_sel].iloc[0]
        source_ref = fila_ref['Source']
        status_ref = fila_ref['Status']
        
        recoms = df_anime[
            (df_anime['Source'] == source_ref) & 
            (df_anime['Status'] == status_ref) & 
            (df_anime['Title'] != anime_sel)
        ].sort_values(by="Score", ascending=False).head(5)
        
        st.subheader(f"Si te gustó '{anime_sel}', te recomendamos revisar:")
        st.table(recoms[['Title', 'Source', 'Status', 'Score']])

# ==========================================================
# OPCIÓN 5: CARGA DE ARCHIVOS
# ==========================================================
elif seleccion == "5. Carga de Archivos":
    st.markdown("<h1 class='main-title'>Carga Externa de Archivos</h1>", unsafe_allow_html=True)
    arch = st.file_uploader("Sube un CSV o Excel:", type=["csv", "xlsx"])
    if arch is not None:
        df_u = pd.read_csv(arch) if arch.name.endswith('.csv') else pd.read_excel(arch)
        st.success("¡Cargado con éxito!")
        st.dataframe(df_u.head(10))
        
        nums = df_u.select_dtypes(include=[np.number]).columns.tolist()
        if nums:
            x_col = st.selectbox("Eje X del archivo subido:", df_u.columns.tolist())
            y_col = st.selectbox("Eje Y numérico:", nums)
            st.plotly_chart(px.bar(df_u.head(30), x=x_col, y=y_col, title="Gráfico Dinámico"), use_container_width=True)

# ==========================================================
# OPCIÓN 6: ANÁLISIS DE SENTIMIENTOS
# ==========================================================
elif seleccion == "6. Análisis de Sentimientos":
    st.markdown("<h1 class='main-title'>Análisis de Sentimientos de la Comunidad</h1>", unsafe_allow_html=True)
    opiniones = pd.DataFrame({
        'Usuario': ['Otaku99', 'SistemasUGB', 'Kira_ElSal', 'AnimeFan'],
        'Comentario': [
            '¡Excelente animación y banda sonora! Un total 10 de 10.',
            'La trama es sumamente lenta y decepcionante comparada con el manga.',
            'Un desarrollo intermedio, aceptable para pasar el rato el fin de semana.',
            'Espectacular giro en la historia, se convirtió en mi favorita.'
        ],
        'Clasificación': ['Positivo', 'Negativo', 'Neutral', 'Positivo']
    })
    st.dataframe(opiniones)
    st.plotly_chart(px.pie(opiniones, names='Clasificación', title='Aceptación en Foros', color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)