# uber_streamlit_app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
import warnings
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import tempfile
import os

# NUEVO: para mapas con fondo sin token
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')

# =========================
# Configuración de la página
# =========================
st.set_page_config(
    page_title="Análisis Uber 2014",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Estilos personalizados
# =========================
def set_custom_style():
    st.markdown(
        """
        <style>
        h1, h2, h3 {
            font-family: "Segoe UI", sans-serif;
        }

        /* Tarjetas de métricas compatibles con tema oscuro/claro */
        .metric-card {
            padding: 1rem 1.2rem;
            border-radius: 0.8rem;
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.35);
        }
        .metric-label {
            font-size: 0.75rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 600;
            color: #e5e7eb;
        }
        .metric-help {
            font-size: 0.75rem;
            color: #6b7280;
        }

        .tab-caption {
            font-size: 0.9rem;
            color: #9ca3af;
            margin-bottom: 0.8rem;
        }

        .dataframe td, .dataframe th {
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_style()

# =========================
# Constantes de franjas horarias
# =========================
MORNING_HOURS = list(range(6, 12))           # 06–11
AFTERNOON_HOURS = list(range(12, 19))        # 12–18
NIGHT_HOURS = list(range(19, 24)) + list(range(0, 6))  # 19–05

# =========================
# Funciones auxiliares (insights, orden)
# =========================
def _orden_dias_en():
    dias_en = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dias_es = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
    mapa_es = dict(zip(dias_en, dias_es))
    return dias_en, dias_es, mapa_es

def _insight_top_bottom(series, etiqueta_unidad="viajes"):
    if series.empty or series.sum() == 0:
        return "No hay datos para los filtros aplicados."
    s = series.fillna(0).astype(int)
    top = s.idxmax(); top_v = int(s.max())
    low = s.idxmin(); low_v = int(s.min())
    rango = top_v - low_v
    return f"Máximo en **{top}** con **{top_v:,} {etiqueta_unidad}**; mínimo en **{low}** con **{low_v:,}**. Diferencial: **{rango:,}**."

def _insight_linea_horas(df_viajes_hora):
    if df_viajes_hora.empty:
        return "No hay datos para los filtros aplicados."
    total_por_hora = df_viajes_hora.sum(axis=1)
    pico_h = int(total_por_hora.idxmax()); pico_v = int(total_por_hora.max())
    valle_h = int(total_por_hora.idxmin()); valle_v = int(total_por_hora.min())
    return (f"Pico agregado a las **{pico_h:02d}:00** con **{pico_v:,} viajes**; "
            f"menor actividad a las **{valle_h:02d}:00** con **{valle_v:,}**.")

def _insight_heatmap(tabla, meses_es, dias_es):
    if tabla.empty:
        return "No hay datos para los filtros aplicados."
    max_pos = np.unravel_index(np.argmax(tabla.values), tabla.shape)
    min_pos = np.unravel_index(np.argmin(tabla.values), tabla.shape)
    max_dia = dias_es[max_pos[0]]; max_mes = meses_es[max_pos[1]]; max_v = int(tabla.values[max_pos])
    min_dia = dias_es[min_pos[0]]; min_mes = meses_es[min_pos[1]]; min_v = int(tabla.values[min_pos])
    return (f"Mayor intensidad en **{max_dia} de {max_mes}** (**{max_v:,} viajes**). "
            f"Menor intensidad en **{min_dia} de {min_mes}** (**{min_v:,}**).")

def _insight_mapa_general(df, sampleado):
    if df is None or df.empty:
        return "No hay datos para los filtros aplicados."
    n = len(df)
    txt = f"Se visualizaron **{n:,} puntos**"
    if sampleado:
        txt += " (muestra aleatoria para mejorar el rendimiento)."
    return txt

def _insight_cuadricula(datasets):
    conteos = {mes: len(df) for mes, df in datasets.items()}
    if not conteos:
        return "No hay datos para los filtros aplicados."
    mes_top = max(conteos, key=conteos.get)
    return f"El mes con mayor densidad de puntos en el mapa es **{mes_top}**."

# === Utilidades bivariadas (lat/lon)
def _filtra_horas(df, horas):
    return df[df['Hora'].isin(horas)].copy()

def _centroide_lat_lon(df):
    if df.empty:
        return np.nan, np.nan
    return df['Lat'].mean(), df['Lon'].mean()

def _hora_pico_y_centroide(df, horas):
    d = _filtra_horas(df, horas)
    if d.empty:
        return None, 0, (np.nan, np.nan)
    conteo = d['Hora'].value_counts().sort_index()
    hora_pico = int(conteo.idxmax())
    d_hora = d[d['Hora'] == hora_pico]
    lat_c, lon_c = _centroide_lat_lon(d_hora)
    return hora_pico, int(conteo.max()), (lat_c, lon_c)

def _hexbin(df, title, gridsize=60):
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Sin datos", ha='center', va='center')
        ax.axis('off')
        return fig
    fig, ax = plt.subplots(figsize=(9, 6))
    hb = ax.hexbin(df['Lon'], df['Lat'], gridsize=gridsize, cmap='viridis', mincnt=1)
    ax.set_title(title)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Viajes")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

def _scatter_corr_lat_lon(df, sample=20000):
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Sin datos", ha='center', va='center')
        ax.axis('off')
        return fig, np.nan
    d = df.sample(min(sample, len(df)), random_state=42) if len(df) > sample else df
    r = np.corrcoef(d['Lat'], d['Lon'])[0, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d['Lon'], d['Lat'], s=4, alpha=0.3)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(f"Lat vs Lon (muestra) — r={r:.3f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, r

def _hist_distribucion(series, label, bins=40, kde=True):
    """Histograma con zoom automático en la zona donde están casi todos los datos."""
    data = series.dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins, alpha=0.7, density=False)

    if kde and len(data) > 1:
        sns.kdeplot(x=data, ax=ax)

    if len(data) > 0:
        q1 = data.quantile(0.01)
        q99 = data.quantile(0.99)
        if q99 > q1:
            margen = (q99 - q1) * 0.15
            ax.set_xlim(q1 - margen, q99 + margen)

    ax.set_title(f"Distribución de {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

# =========================
# Cargar datos
# =========================
@st.cache_data
def load_data():
    try:
        df_abril = pd.read_csv('uber-raw-data-apr14.csv')
        df_mayo = pd.read_csv('uber-raw-data-may14.csv')
        df_junio = pd.read_csv('uber-raw-data-jun14.csv')
        df_julio = pd.read_csv('uber-raw-data-jul14.csv')
        df_agosto = pd.read_csv('uber-raw-data-aug14.csv')
        df_septiembre = pd.read_csv('uber-raw-data-sep14.csv')

        datasets = {
            'Abril': df_abril,
            'Mayo': df_mayo,
            'Junio': df_junio,
            'Julio': df_julio,
            'Agosto': df_agosto,
            'Septiembre': df_septiembre
        }

        def preparar_datos(df, mes_nombre):
            df = df.copy()
            df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%m/%d/%Y %H:%M:%S')
            df['Mes'] = mes_nombre
            df['Hora'] = df['Date/Time'].dt.hour
            df['Dia_Semana'] = df['Date/Time'].dt.day_name()
            df['Dia_Mes'] = df['Date/Time'].dt.day
            return df

        datasets_preparados = {n: preparar_datos(d, n) for n, d in datasets.items()}
        df_completo = pd.concat(datasets_preparados.values(), ignore_index=True)
        return datasets_preparados, df_completo
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None, None

# =========================
# Guardar plot temporal (para PDF)
# =========================
def save_plot_to_temp(fig, filename):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)
    fig.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return temp_path

# =========================
# Reporte PDF
# =========================
def generate_pdf_report(datasets, df_completo, meses_seleccionados, bases_seleccionadas, 
                       fig_heatmap=None, fig_mapa_general=None, fig_cuadricula=None,
                       tabla_preguntas=None,
                       fig_hex_m=None, fig_hex_t=None, fig_hex_n=None,
                       fig_scatter=None, fig_hist_lat=None, fig_hist_lon=None):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("REPORTE DE ANÁLISIS UBER 2014", title_style))

    filtros_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray,
        alignment=1
    )
    meses_text = ", ".join(meses_seleccionados) if meses_seleccionados else "Todos los meses"
    bases_text = ", ".join(bases_seleccionadas) if bases_seleccionadas else "Todas las bases"
    story.append(Paragraph(f"Filtros aplicados: Meses: {meses_text} | Bases: {bases_text}", filtros_style))
    story.append(Spacer(1, 12))

    # Resumen por mes
    section_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12
    )
    story.append(Paragraph("ESTADÍSTICAS RESUMEN", section_style))
    resumen_mes = []
    for mes, df in datasets.items():
        total = len(df)
        bases = df['Base'].nunique()
        fecha_min = df['Date/Time'].min().strftime('%d/%m/%Y')
        fecha_max = df['Date/Time'].max().strftime('%d/%m/%Y')
        resumen_mes.append([mes, f"{total:,}", bases, fecha_min, fecha_max])

    resumen_data = [['Mes', 'Total Viajes', 'Bases', 'Fecha Inicio', 'Fecha Fin']] + resumen_mes
    resumen_table = Table(resumen_data, colWidths=[80, 80, 50, 80, 80])
    resumen_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(resumen_table)
    story.append(Spacer(1, 18))

    # Bloque de preguntas (tabla)
    if tabla_preguntas is not None:
        story.append(Paragraph("RESPUESTAS CLAVE (Mañana, Tarde, Noche–Madrugada)", section_style))
        data = [['Franja', 'Hora Pico', 'Viajes Pico', 'Centroide Lat', 'Centroide Lon']]
        for fila in tabla_preguntas:
            data.append([
                fila['Franja'],
                fila['Hora Pico'],
                f"{fila['Viajes Pico']:,}",
                f"{fila['Centroide Lat']:.4f}",
                f"{fila['Centroide Lon']:.4f}"
            ])
        t = Table(data, colWidths=[90, 70, 80, 90, 90])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0B7285')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA'))
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    # Función interna para añadir figuras
    def _add_fig(fig, name, w, h):
        if fig is None:
            return
        path = save_plot_to_temp(fig, name)
        story.append(Image(path, width=w, height=h))
        story.append(Spacer(1, 10))

    # Figuras bivariadas
    _add_fig(fig_hex_m, "hex_manan.pdf.png", 380, 230)
    _add_fig(fig_hex_t, "hex_tarde.pdf.png", 380, 230)
    _add_fig(fig_hex_n, "hex_noche.pdf.png", 380, 230)
    _add_fig(fig_scatter, "scatter_lat_lon.pdf.png", 380, 230)
    _add_fig(fig_hist_lat, "hist_lat.pdf.png", 380, 230)
    _add_fig(fig_hist_lon, "hist_lon.pdf.png", 380, 230)

    # Heatmap y mapas
    if fig_heatmap is not None:
        story.append(Paragraph("MAPA DE CALOR - VIAJES POR DÍA Y MES", section_style))
        path = save_plot_to_temp(fig_heatmap, "heatmap.png")
        story.append(Image(path, width=400, height=250))
        story.append(Spacer(1, 14))

    if fig_mapa_general is not None:
        story.append(Paragraph("MAPA GENERAL DE UBICACIONES", section_style))
        path = save_plot_to_temp(fig_mapa_general, "mapa_general.png")
        story.append(Image(path, width=400, height=250))
        story.append(Spacer(1, 10))

    if fig_cuadricula is not None:
        story.append(Paragraph("MAPAS POR MES - VISTA EN CUADRÍCULA", section_style))
        path = save_plot_to_temp(fig_cuadricula, "mapa_cuadricula.png")
        story.append(Image(path, width=400, height=250))

    # Pie
    story.append(Spacer(1, 16))
    filtros_style_small = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.gray,
        alignment=1
    )
    story.append(Paragraph(f"Reporte generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", filtros_style_small))

    doc.build(story)
    buffer.seek(0)
    return buffer

# =========================
# Visualizaciones base (matplotlib, para PDF)
# =========================
def crear_mapa_calor_corregido(df):
    meses_orden_eng = ["April", "May", "June", "July", "August", "September"]
    meses_orden_esp = ["Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre"]
    dias_orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dias_orden_esp = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    mapeo_meses = {
        'Abril': 'April',
        'Mayo': 'May',
        'Junio': 'June',
        'Julio': 'July',
        'Agosto': 'August',
        'Septiembre': 'September'
    }

    df_heatmap = df.copy()
    df_heatmap['Month_Eng'] = df_heatmap['Mes'].map(mapeo_meses)
    tabla = (
        df_heatmap
        .assign(
            Month_Eng=pd.Categorical(df_heatmap['Month_Eng'], categories=meses_orden_eng, ordered=True),
            Dia_Semana=pd.Categorical(df_heatmap['Dia_Semana'], categories=dias_orden, ordered=True)
        )
        .pivot_table(index='Dia_Semana', columns='Month_Eng', values='Date/Time', aggfunc='count')
        .reindex(index=dias_orden, columns=meses_orden_eng)
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(tabla.values, aspect='auto', cmap='Blues')
    ax.set_xticks(np.arange(len(meses_orden_esp)))
    ax.set_xticklabels(meses_orden_esp)
    ax.set_yticks(np.arange(len(dias_orden_esp)))
    ax.set_yticklabels(dias_orden_esp)
    ax.set_xlabel('Mes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Día de la Semana', fontsize=12, fontweight='bold')
    ax.set_title('Mapa de calor: viajes por día de la semana y mes (filtros aplicados)', fontsize=14, fontweight='bold', pad=20)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Número de Viajes', rotation=-90, va="bottom", fontsize=10)

    for i in range(len(tabla.index)):
        for j in range(len(tabla.columns)):
            ax.text(j, i, f'{tabla.iloc[i, j]:,}', ha="center", va="center",
                    color="black", fontsize=10, fontweight='bold')

    ax.set_xticks(np.arange(-0.5, len(tabla.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tabla.index), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    plt.tight_layout()
    return fig, tabla

def crear_mapa_ubicaciones_cuadricula_matplotlib(df_completo, datasets, max_puntos=10000):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    meses_orden = ['Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre']

    for i, mes in enumerate(meses_orden):
        if mes in datasets:
            df_mes = datasets[mes]
            df_plot = df_mes.sample(max_puntos, random_state=42) if len(df_mes) > max_puntos else df_mes
            scatter = axes[i].scatter(
                df_plot['Lon'], df_plot['Lat'],
                s=1, alpha=0.4,
                c=df_plot['Date/Time'].dt.hour,
                cmap='viridis'
            )
            axes[i].set_title(f'{mes} 2014\n({len(df_plot):,} puntos)', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Longitud', fontsize=10)
            axes[i].set_ylabel('Latitud', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=axes[i], shrink=0.8)
            cbar.set_label('Hora', rotation=270, labelpad=10, fontsize=8)

    for i in range(len(meses_orden), 6):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig

def crear_mapa_general_matplotlib(df_completo, max_puntos=20000):
    sampleado = False
    if len(df_completo) > max_puntos:
        df_muestra = df_completo.sample(max_puntos, random_state=42)
        sampleado = True
    else:
        df_muestra = df_completo

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        df_muestra['Lon'], df_muestra['Lat'],
        s=1, alpha=0.3,
        c=df_muestra['Date/Time'].dt.hour,
        cmap='viridis'
    )
    ax.set_title('Mapa General - Todas las Ubicaciones de Viajes Uber\n(Abril-Septiembre 2014)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hora del Día', rotation=270, labelpad=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, df_muestra, sampleado

# =========================
# Mapas interactivos con fondo (Folium + OpenStreetMap)
# =========================
def render_folium_map(m, height=600):
    """Renderiza un mapa de Folium dentro de Streamlit."""
    components.html(m._repr_html_(), height=height, scrolling=False)

def mostrar_mapa_general_ny(df_completo, max_puntos=20000):
    sampleado = False
    if len(df_completo) > max_puntos:
        df_muestra = df_completo.sample(max_puntos, random_state=42)
        sampleado = True
    else:
        df_muestra = df_completo

    # centro de NYC
    m = folium.Map(location=[40.75, -73.98], zoom_start=11, tiles="CartoDB positron")

    HeatMap(
        data=df_muestra[['Lat', 'Lon']].values.tolist(),
        radius=6,
        blur=4,
        max_zoom=13
    ).add_to(m)

    render_folium_map(m, height=600)
    return df_muestra, sampleado

def mostrar_mapas_por_mes_ny(datasets, max_puntos=10000):
    filas = [
        ['Abril', 'Mayo', 'Junio'],
        ['Julio', 'Agosto', 'Septiembre']
    ]
    for fila in filas:
        cols = st.columns(len(fila))
        for col, mes in zip(cols, fila):
            if mes in datasets:
                with col:
                    st.markdown(f"**{mes} 2014**")
                    df_mes = datasets[mes]
                    df_plot = df_mes.sample(max_puntos, random_state=42) if len(df_mes) > max_puntos else df_mes
                    m = folium.Map(location=[40.75, -73.98], zoom_start=11, tiles="CartoDB positron")
                    HeatMap(
                        data=df_plot[['Lat', 'Lon']].values.tolist(),
                        radius=6,
                        blur=4,
                        max_zoom=13
                    ).add_to(m)
                    render_folium_map(m, height=400)

# =========================
# Interfaz principal
# =========================
def main():
    # Encabezado principal
    st.title("🚗 Análisis de Datos Uber 2014")
    st.markdown(
        """
        Observa el comportamiento de los viajes de Uber en Nueva York entre **abril y septiembre de 2014**.
        Usa los filtros de la izquierda para responder preguntas como:
        - ¿Qué bases son más activas?
        - ¿En qué horarios hay más viajes?
        - ¿Cómo se distribuyen los puntos geográficos en la ciudad?
        """
    )
    st.markdown("---")

    # Carga de datos
    with st.spinner("Cargando datos y preparando todo..."):
        datasets, df_completo = load_data()
    if datasets is None:
        st.error("No se pudieron cargar los datos. Verifica que los archivos CSV estén en el directorio.")
        return

    # Sidebar de filtros
    st.sidebar.title("🎛️ Filtros")
    st.sidebar.caption("Ajusta los filtros y observa cómo cambian las visualizaciones.")

    meses_disponibles = list(datasets.keys())
    meses_seleccionados = st.sidebar.multiselect(
        "Seleccionar meses:",
        meses_disponibles,
        default=meses_disponibles
    )

    todas_bases = sorted(df_completo['Base'].unique())
    bases_seleccionadas = st.sidebar.multiselect(
        "Seleccionar bases:",
        todas_bases,
        default=todas_bases
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("📄 Cuando tengas listo tu análisis, genera el reporte en PDF al final.")

    # DF filtrado global
    df_filtrado = df_completo.copy()
    if meses_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['Mes'].isin(meses_seleccionados)]
    if bases_seleccionadas:
        df_filtrado = df_filtrado[df_filtrado['Base'].isin(bases_seleccionadas)]

    # =========================
    # Métricas principales (tarjetas)
    # =========================
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Viajes totales (dataset completo)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(df_completo):,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-help">Antes de aplicar filtros</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Bases únicas</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{df_completo["Base"].nunique()}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-help">Número de bases registradas</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        periodo = f"{df_completo['Date/Time'].min().strftime('%d/%m/%Y')} – {df_completo['Date/Time'].max().strftime('%d/%m/%Y')}"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Período analizado</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{periodo}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-help">Rango de fechas en los datos</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        dias_totales = (df_completo['Date/Time'].max() - df_completo['Date/Time'].min()).days
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Días analizados</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{dias_totales}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-help">Desde el primer hasta el último viaje</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.caption(f"🔍 Con los filtros actuales se están analizando **{len(df_filtrado):,} viajes**.")
    st.markdown("---")

    # =========================
    # Pestañas principales
    # =========================
    tab_overview, tab_bases, tab_tiempo, tab_mapas, tab_latlon, tab_pdf = st.tabs([
        "📌 Visión general",
        "🏢 Bases",
        "⏰ Patrones temporales",
        "🗺️ Mapas",
        "📍 Latitudes y longitudes",
        "📄 Reporte PDF"
    ])

    # ---------- TAB 1: Visión general ----------
    with tab_overview:
        st.markdown('<p class="tab-caption">Resumen rápido del dataset por mes y día de la semana.</p>', unsafe_allow_html=True)

        st.subheader("Resumen por mes")
        resumen_mes = []
        for mes, df in datasets.items():
            resumen_mes.append({
                'Mes': mes,
                'Total Viajes': len(df),
                'Bases': df['Base'].nunique(),
                'Fecha Inicio': df['Date/Time'].min().strftime('%d/%m/%Y'),
                'Fecha Fin': df['Date/Time'].max().strftime('%d/%m/%Y')
            })
        st.dataframe(
            pd.DataFrame(resumen_mes).style.format({'Total Viajes': '{:,}', 'Bases': '{}'}),
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("Viajes por día de la semana (con filtros)")
        dias_en, dias_es, _ = _orden_dias_en()
        conteo_dow = (
            df_filtrado['Dia_Semana'].value_counts()
            .reindex(dias_en)
            .fillna(0)
            .astype(int)
        )

        fig_dow, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(dias_es)), conteo_dow.values)
        ax.set_xticks(range(len(dias_es)))
        ax.set_xticklabels(dias_es)
        ax.set_ylabel('Viajes')
        ax.set_title('Viajes por día (filtros aplicados)')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_dow)
        st.markdown(f"**¿Qué vemos?** {_insight_top_bottom(pd.Series(conteo_dow.values, index=dias_es))}")

    # ---------- TAB 2: Bases ----------
    with tab_bases:
        st.markdown('<p class="tab-caption">Compara la actividad de las bases seleccionadas por mes.</p>', unsafe_allow_html=True)
        st.subheader("Comparación de actividad por base")

        if meses_seleccionados and bases_seleccionadas:
            datos_comparativos = {}
            for mes in meses_seleccionados:
                df_mes = datasets[mes]
                df_mes = df_mes[df_mes['Base'].isin(bases_seleccionadas)]
                datos_comparativos[mes] = df_mes['Base'].value_counts()

            df_comparacion = pd.DataFrame(datos_comparativos).fillna(0)

            fig_cmp, ax = plt.subplots(figsize=(12, 6))
            df_comparacion.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'Comparación por Base ({", ".join(meses_seleccionados)})')
            ax.set_xlabel('Base')
            ax.set_ylabel('Viajes')
            ax.legend(title='Mes')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_cmp)

            totales = df_comparacion.sum(axis=1).sort_values(ascending=False)
            if not totales.empty:
                st.markdown(
                    f"**¿Qué vemos?** La base con mayor actividad es **{totales.index[0]}** "
                    f"con **{int(totales.iloc[0]):,} viajes**. "
                    f"{_insight_top_bottom(totales, 'viajes totales')}"
                )

            st.subheader("Datos detallados")
            st.dataframe(df_comparacion.style.format("{:,.0f}"), use_container_width=True)
        else:
            st.info("Selecciona al menos **un mes** y **una base** en la barra lateral para ver la comparación.")

    # ---------- TAB 3: Patrones temporales ----------
    with tab_tiempo:
        st.markdown('<p class="tab-caption">Explora cómo cambian los viajes según la hora del día y la combinación día × mes.</p>', unsafe_allow_html=True)

        st.subheader("Viajes por hora del día")
        if meses_seleccionados:
            datos_hora = {}
            for mes in meses_seleccionados:
                df_mes = datasets[mes]
                if bases_seleccionadas:
                    df_mes = df_mes[df_mes['Base'].isin(bases_seleccionadas)]
                datos_hora[mes] = df_mes['Hora'].value_counts().sort_index()

            df_viajes_hora = pd.DataFrame(datos_hora).fillna(0).astype(int)
            fig_hora, ax = plt.subplots(figsize=(12, 6))
            for mes in df_viajes_hora.columns:
                ax.plot(df_viajes_hora.index, df_viajes_hora[mes], marker='o', label=mes, linewidth=2)
            ax.set_title('Viajes por hora (filtros aplicados)')
            ax.set_xlabel('Hora')
            ax.set_ylabel('Viajes')
            ax.set_xticks(range(0, 24))
            ax.legend(title='Mes')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_hora)
            st.markdown(f"**¿Qué vemos?** {_insight_linea_horas(df_viajes_hora)}")
        else:
            st.info("Selecciona al menos un **mes** para ver la curva por hora.")

        st.markdown("---")

        st.subheader("Mapa de calor: Día de la semana × Mes")
        fig_heatmap, tabla_heatmap = crear_mapa_calor_corregido(df_filtrado)
        st.pyplot(fig_heatmap)
        meses_es = ["Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre"]
        st.markdown(
            f"**¿Qué vemos?** {_insight_heatmap(tabla_heatmap, meses_es, ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'])}"
        )

    # ---------- TAB 4: Mapas (con fondo de NY) ----------
    with tab_mapas:
        st.markdown('<p class="tab-caption">Visualiza la distribución espacial de los viajes directamente sobre un mapa de Nueva York.</p>', unsafe_allow_html=True)

        st.subheader("Mapa general de ubicaciones (fondo: Nueva York)")
        df_muestra, sampleado = mostrar_mapa_general_ny(df_completo)
        st.markdown(f"**¿Qué vemos?** {_insight_mapa_general(df_muestra, sampleado)}")

        st.markdown("---")

        st.subheader("Mapas por mes (sobre el mismo mapa de la ciudad)")
        mostrar_mapas_por_mes_ny(datasets)
        st.markdown(f"**Nota:** Todos los puntos se proyectan sobre el área de Nueva York, donde ocurrieron estos viajes.")

    # ---------- TAB 5: Latitudes y longitudes ----------
    with tab_latlon:
        st.markdown(
            '<p class="tab-caption">Analiza cómo se distribuyen las coordenadas (latitud y longitud) y cómo cambian según la franja horaria.</p>',
            unsafe_allow_html=True
        )

        st.subheader("Franjas horarias (Mañana, Tarde, Noche–Madrugada)")
        st.caption("Mañana = 06–11, Tarde = 12–18, Noche–Madrugada = 19–05")

        colA, colB, colC = st.columns(3)
        h_m, v_m, (lat_m, lon_m) = _hora_pico_y_centroide(df_filtrado, MORNING_HOURS)
        h_t, v_t, (lat_t, lon_t) = _hora_pico_y_centroide(df_filtrado, AFTERNOON_HOURS)
        h_n, v_n, (lat_n, lon_n) = _hora_pico_y_centroide(df_filtrado, NIGHT_HOURS)

        with colA:
            st.markdown("### Mañana (06–11)")
            st.write("Hora pico:", f"**{h_m:02d}:00**" if h_m is not None else "–")
            st.write("Viajes pico:", f"**{v_m:,}**" if v_m else "–")
            st.write("Centroide:", f"**({lat_m:.4f}, {lon_m:.4f})**" if h_m is not None else "–")

        with colB:
            st.markdown("### Tarde (12–18)")
            st.write("Hora pico:", f"**{h_t:02d}:00**" if h_t is not None else "–")
            st.write("Viajes pico:", f"**{v_t:,}**" if v_t else "–")
            st.write("Centroide:", f"**({lat_t:.4f}, {lon_t:.4f})**" if h_t is not None else "–")

        with colC:
            st.markdown("### Noche–Madrugada (19–05)")
            st.write("Hora pico:", f"**{h_n:02d}:00**" if h_n is not None else "–")
            st.write("Viajes pico:", f"**{v_n:,}**" if v_n else "–")
            st.write("Centroide:", f"**({lat_n:.4f}, {lon_n:.4f})**" if h_n is not None else "–")

        st.markdown("---")

        st.subheader("Mapas de concentración por franja (hexbin)")
        gridsize = st.slider("Resolución de los hexágonos", 20, 100, 60, 5)
        fig_hex_m = _hexbin(_filtra_horas(df_filtrado, MORNING_HOURS), "Mañana 06–11", gridsize)
        fig_hex_t = _hexbin(_filtra_horas(df_filtrado, AFTERNOON_HOURS), "Tarde 12–18", gridsize)
        fig_hex_n = _hexbin(_filtra_horas(df_filtrado, NIGHT_HOURS), "Noche–Madrugada 19–05", gridsize)

        c1h, c2h, c3h = st.columns(3)
        with c1h:
            st.pyplot(fig_hex_m)
            st.caption("La densidad de puntos muestra dónde se concentran los viajes en la mañana.")
        with c2h:
            st.pyplot(fig_hex_t)
            st.caption("Aquí se ve cómo se desplaza la concentración de viajes durante la tarde.")
        with c3h:
            st.pyplot(fig_hex_n)
            st.caption("Patrones de actividad nocturna y de madrugada en la ciudad.")

        st.markdown("---")

        st.subheader("Relación entre latitud y longitud")
        fig_scatter, r = _scatter_corr_lat_lon(df_filtrado, sample=25000)
        st.pyplot(fig_scatter)
        st.write("**r de Pearson**:", "N/A" if np.isnan(r) else f"**{r:.3f}**")
        st.caption(
            "Cada punto representa una ubicación de viaje. La nube de puntos dibuja la forma aproximada de Nueva York. "
            "El valor r de Pearson indica qué tan alineadas están las coordenadas (solo como medida geométrica)."
        )

        st.markdown("---")

        st.subheader("Distribuciones de latitudes y longitudes (zoom a la zona útil)")
        fig_hist_lat = _hist_distribucion(df_filtrado['Lat'], "Latitud", bins=50)
        fig_hist_lon = _hist_distribucion(df_filtrado['Lon'], "Longitud", bins=50)
        cdl, cdo = st.columns(2)
        with cdl:
            st.pyplot(fig_hist_lat)
            st.caption(
                "Esta gráfica muestra cómo se distribuyen las latitudes de los viajes. "
                "El pico central indica el rango de latitudes donde se concentran casi todos los puntos (zona urbana de Nueva York)."
            )
        with cdo:
            st.pyplot(fig_hist_lon)
            st.caption(
                "Esta gráfica muestra la distribución de las longitudes. "
                "El bloque central corresponde a la franja este-oeste donde se ubican la mayoría de los viajes."
            )

        tabla_preguntas = [
            {
                'Franja': 'Mañana',
                'Hora Pico': f"{h_m:02d}:00" if h_m is not None else '–',
                'Viajes Pico': v_m,
                'Centroide Lat': (lat_m if h_m is not None else np.nan),
                'Centroide Lon': (lon_m if h_m is not None else np.nan)
            },
            {
                'Franja': 'Tarde',
                'Hora Pico': f"{h_t:02d}:00" if h_t is not None else '–',
                'Viajes Pico': v_t,
                'Centroide Lat': (lat_t if h_t is not None else np.nan),
                'Centroide Lon': (lon_t if h_t is not None else np.nan)
            },
            {
                'Franja': 'Noche–Madrugada',
                'Hora Pico': f"{h_n:02d}:00" if h_n is not None else '–',
                'Viajes Pico': v_n,
                'Centroide Lat': (lat_n if h_n is not None else np.nan),
                'Centroide Lon': (lon_n if h_n is not None else np.nan)
            }
        ]

        st.session_state['tabla_preguntas'] = tabla_preguntas
        st.session_state['fig_hex_m'] = fig_hex_m
        st.session_state['fig_hex_t'] = fig_hex_t
        st.session_state['fig_hex_n'] = fig_hex_n
        st.session_state['fig_scatter'] = fig_scatter
        st.session_state['fig_hist_lat'] = fig_hist_lat
        st.session_state['fig_hist_lon'] = fig_hist_lon

    # ---------- TAB 6: PDF ----------
    with tab_pdf:
        st.markdown('<p class="tab-caption">Genera un reporte completo en PDF con las principales visualizaciones e insights.</p>', unsafe_allow_html=True)
        st.subheader("Generar reporte PDF")

        fig_heatmap, tabla_heatmap = crear_mapa_calor_corregido(df_filtrado)
        fig_mapa_general, df_muestra_pdf, sampleado_pdf = crear_mapa_general_matplotlib(df_completo)
        fig_cuadricula = crear_mapa_ubicaciones_cuadricula_matplotlib(df_completo, datasets)

        tabla_preguntas = st.session_state.get('tabla_preguntas', None)
        fig_hex_m = st.session_state.get('fig_hex_m', None)
        fig_hex_t = st.session_state.get('fig_hex_t', None)
        fig_hex_n = st.session_state.get('fig_hex_n', None)
        fig_scatter = st.session_state.get('fig_scatter', None)
        fig_hist_lat = st.session_state.get('fig_hist_lat', None)
        fig_hist_lon = st.session_state.get('fig_hist_lon', None)

        st.info("El reporte respetará los filtros actuales de **meses** y **bases**.")

        if st.button("📥 Generar reporte completo en PDF"):
            with st.spinner("Generando reporte PDF..."):
                pdf_buffer = generate_pdf_report(
                    datasets=datasets,
                    df_completo=df_completo,
                    meses_seleccionados=meses_seleccionados,
                    bases_seleccionadas=bases_seleccionadas,
                    fig_heatmap=fig_heatmap,
                    fig_mapa_general=fig_mapa_general,
                    fig_cuadricula=fig_cuadricula,
                    tabla_preguntas=tabla_preguntas,
                    fig_hex_m=fig_hex_m,
                    fig_hex_t=fig_hex_t,
                    fig_hex_n=fig_hex_n,
                    fig_scatter=fig_scatter,
                    fig_hist_lat=fig_hist_lat,
                    fig_hist_lon=fig_hist_lon
                )
                st.success("¡Reporte generado exitosamente!")
                st.download_button(
                    label="Descargar Reporte PDF",
                    data=pdf_buffer,
                    file_name=f"reporte_uber_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

# Ejecutar la aplicación
if __name__ == "__main__":
    main()


