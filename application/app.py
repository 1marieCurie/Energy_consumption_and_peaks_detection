import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from scipy.signal import find_peaks
from sklearn.metrics import r2_score
import os

st.set_page_config(page_title="Prédiction Énergétique", layout="wide", initial_sidebar_state="expanded")

# -------------------------------------------------------
# CSS personnalisé pour le design (adapté au background dark)
# -------------------------------------------------------
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main-container {
        border-radius: 15px;
        padding: 40px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        text-align: center;
        transition: all 0.3s ease;
        animation: slideInUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
    }
    
    .metric-value {
        font-size: 2.8em;
        font-weight: 700;
        margin: 15px 0;
        letter-spacing: -1px;
        color: #ffffff;
    }
    
    .metric-label {
        font-size: 0.95em;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        color: #ffffff;
    }
    
    .section-title {
        font-size: 1.8em;
        font-weight: 700;
        color: #ffffff;
        margin: 35px 0 20px 0;
        padding-bottom: 15px;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    .upload-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
        border-color: #764ba2;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.25);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-left: 4px solid #667eea;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        color: #ffffff;
    }
    
    .info-box h4 {
        color: #667eea;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    .info-box p, .info-box ul, .info-box li {
        color: #e0e0e0;
    }
    
    .success-message {
        animation: fadeInDown 0.5s ease-out;
        color: #10b981;
        font-weight: 600;
        padding: 10px 15px;
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10b981;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    .chart-container {
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        margin: 20px 0;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .dataframe-wrapper {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        margin: 20px 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
    }
    
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Chargement des objets sauvegardés
# -------------------------------------------------------
@st.cache_resource
def load_objects():
    folder = "Projet_ML_Electricity_Consumption"
    required_files = ["scaler.pkl", "selector.pkl", "selected_features.pkl", "ridge_model.pkl"]
    for f in required_files:
        path = os.path.join(folder, f)
        if not os.path.exists(path):
            st.error(f"Le fichier {f} est manquant dans le dossier {folder}.")
            st.stop()
    with open(os.path.join(folder, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(folder, "selector.pkl"), "rb") as f:
        selector = pickle.load(f)
    with open(os.path.join(folder, "selected_features.pkl"), "rb") as f:
        selected_features = pickle.load(f)
    with open(os.path.join(folder, "ridge_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return scaler, selector, selected_features, model

scaler, selector, selected_features, model = load_objects()

# -------------------------------------------------------
# Liste complète des 32 features
# -------------------------------------------------------
features = [
    'year','month','day_of_week','hour','is_weekend','n_occupants','is_peak',
    'hour_day_interaction','month_day_interaction',
    'y_kWh_lag1','y_kWh_lag2','lag_24','lag_48','lag_168','lag_336',
    'rolling_6h','rolling_24h','rolling_7d',
    'std_24h','expanding_mean','expanding_std',
    'diff_1h','diff_24h','rolling_24h_pct_change',
    'hour_n_occupants','dow_weekend',
    'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos'
]

# -------------------------------------------------------
# Générer toutes les features
# -------------------------------------------------------
def prepare_features(df):
    df = df.copy()

    df["hour_day_interaction"] = df["hour"] * df["day_of_week"]
    df["month_day_interaction"] = df["month"] * df["day_of_week"]
    df["hour_n_occupants"] = df["hour"] * df["n_occupants"]
    df["dow_weekend"] = df["day_of_week"] * df["is_weekend"]

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"] = np.sin(2*np.pi*df["day_of_week"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["day_of_week"]/7)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

    for col in ['y_kWh_lag1','y_kWh_lag2','lag_24','lag_48','lag_168','lag_336',
                'rolling_6h','rolling_24h','rolling_7d','std_24h','expanding_mean','expanding_std',
                'diff_1h','diff_24h','rolling_24h_pct_change']:
        df[col] = 0

    if 'y_kWh' in df.columns:
        df['y_kWh_lag1'] = df['y_kWh'].shift(1).fillna(0)
        df['y_kWh_lag2'] = df['y_kWh'].shift(2).fillna(0)
        df['lag_24'] = df['y_kWh'].shift(24).fillna(0)
        df['lag_48'] = df['y_kWh'].shift(48).fillna(0)
        df['lag_168'] = df['y_kWh'].shift(168).fillna(0)
        df['lag_336'] = df['y_kWh'].shift(336).fillna(0)

        df['rolling_6h'] = df['y_kWh'].rolling(window=6, min_periods=1).mean()
        df['rolling_24h'] = df['y_kWh'].rolling(window=24, min_periods=1).mean()
        df['rolling_7d'] = df['y_kWh'].rolling(window=168, min_periods=1).mean()
        df['std_24h'] = df['y_kWh'].rolling(window=24, min_periods=1).std().fillna(0)
        df['expanding_mean'] = df['y_kWh'].expanding(min_periods=1).mean()
        df['expanding_std'] = df['y_kWh'].expanding(min_periods=1).std().fillna(0)
        df['diff_1h'] = df['y_kWh'].diff().fillna(0)
        df['diff_24h'] = df['y_kWh'].diff(24).fillna(0)
        df['rolling_24h_pct_change'] = df['rolling_24h'].pct_change().fillna(0)

    df = df.reindex(columns=features, fill_value=0)
    return df

# -------------------------------------------------------
# Prédiction
# -------------------------------------------------------
def predict_ridge(model, scaler, selector, df_features):
    X_scaled = scaler.transform(df_features)
    X_selected = selector.transform(X_scaled)
    return model.predict(X_selected)

# -------------------------------------------------------
# En-tête principal
# -------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <h1 style='text-align: center; color: #667eea; margin-bottom: 10px; font-size: 2.5em;'>
        Prédiction Énergétique
    </h1>
    <p style='text-align: center; color: #a0aec0; font-size: 1.1em; font-weight: 500;'>
        Analyse intelligente de la consommation électrique
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------
# Section Instructions
# -------------------------------------------------------
with st.expander("Instructions d'utilisation", expanded=False):
    st.markdown("""
    <div class='info-box'>
    <h4>Colonnes essentielles requises</h4>
    <p>Votre fichier CSV doit contenir les colonnes suivantes :</p>
    <ul>
        <li><strong>year</strong> : Année</li>
        <li><strong>month</strong> : Mois (1-12)</li>
        <li><strong>day_of_week</strong> : Jour de la semaine (0-6)</li>
        <li><strong>hour</strong> : Heure (0-23)</li>
        <li><strong>is_weekend</strong> : Indicateur weekend (0 ou 1)</li>
        <li><strong>n_occupants</strong> : Nombre d'occupants</li>
        <li><strong>is_peak</strong> : Heures de pointe (0 ou 1)</li>
    </ul>
    
    <h4>Colonne optionnelle recommandée</h4>
    <p><strong>y_kWh</strong> : Consommation réelle (au moins 1 semaine recommandée pour les lags)</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# Upload CSV
# -------------------------------------------------------
st.markdown("<div class='section-title'>Importer vos données</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    with st.spinner("Chargement et traitement des données..."):
        df_input = pd.read_csv(uploaded_file)
        st.markdown("<div class='success-message'>Fichier chargé avec succès</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='section-title'>Aperçu des données</div>", unsafe_allow_html=True)
        st.dataframe(df_input.head(10), use_container_width=True)

        # Préparer les features
        df_features = prepare_features(df_input)

        # Prédiction
        df_input["pred"] = predict_ridge(model, scaler, selector, df_features)
        y_pred = df_input["pred"].values

        # Détection des pics sur la prédiction
        if y_pred.size > 1 and (y_pred.max() - y_pred.min()) > 0:
            y_pred_range = y_pred.max() - y_pred.min()
            peaks_pred, _ = find_peaks(y_pred, prominence=y_pred_range * 0.1)
        else:
            peaks_pred = np.array([], dtype=int)
        if y_pred.size > 1:
            if y_pred[0] > y_pred[1]: peaks_pred = np.insert(peaks_pred, 0, 0)
            if y_pred[-1] > y_pred[-2]: peaks_pred = np.append(peaks_pred, len(y_pred) - 1)

        # Pics réels
        peaks_real = np.array([], dtype=int)
        if 'y_kWh' in df_input.columns:
            y_input = df_input["y_kWh"].values
            if y_input.size > 1 and (y_input.max() - y_input.min()) > 0:
                y_input_range = y_input.max() - y_input.min()
                peaks_real, _ = find_peaks(y_input, prominence=y_input_range * 0.1)
                if y_input[0] > y_input[1]: peaks_real = np.insert(peaks_real, 0, 0)
                if y_input[-1] > y_input[-2]: peaks_real = np.append(peaks_real, len(y_input)-1)
            df_input["y_kWh_imported"] = y_input

        # Colonnes pics
        y_peak_pred = np.zeros_like(y_pred, dtype=int)
        if peaks_pred.size: y_peak_pred[peaks_pred] = 1
        df_input["peak_pred"] = y_peak_pred

        if "y_kWh_imported" in df_input.columns:
            y_peak_real = np.zeros_like(df_input["y_kWh_imported"].values, dtype=int)
            if peaks_real.size: y_peak_real[peaks_real] = 1
            df_input["peak_real"] = y_peak_real

        # -------------------------------------------------------
        # Métriques principales
        # -------------------------------------------------------
        st.markdown("<div class='section-title'>Métriques de performance</div>", unsafe_allow_html=True)
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Pics détectés (Prédiction)</div>
                <div class='metric-value'>{len(peaks_pred)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            if "y_kWh_imported" in df_input.columns:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Pics réels (CSV)</div>
                    <div class='metric-value'>{len(peaks_real)}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with metrics_col3:
            if "y_kWh_imported" in df_input.columns:
                r2 = r2_score(df_input["y_kWh_imported"], df_input["pred"])
                r2_color = "#10b981" if r2 > 0.7 else "#f59e0b" if r2 > 0.5 else "#ef4444"
                st.markdown(f"""
                <div class='metric-card' style='background: linear-gradient(135deg, {r2_color} 0%, {r2_color}cc 100%);'>
                    <div class='metric-label'>Taux de confiance (R²)</div>
                    <div class='metric-value'>{r2:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

        # -------------------------------------------------------
        # Graphique prédiction avec filtres
        # -------------------------------------------------------
        st.markdown("<div class='section-title'>Graphique de prédiction</div>", unsafe_allow_html=True)
        
        # Sélection du type d'affichage avec radio buttons
        view_option = st.radio(
            "Sélectionner l'affichage",
            options=["Mois complet", "Par semaine", "Par jour"],
            horizontal=True,
            index=0
        )
        
        # Déterminer la plage de données à afficher
        if view_option == "Par jour":
            # Calculer le nombre de jours disponibles
            nb_jours = max(1, len(df_input) // 24)
            jour_selectione = st.slider(
                "Sélectionner un jour",
                min_value=0,
                max_value=max(0, nb_jours - 1),
                value=0
            )
            # Filtrer les données pour un jour spécifique
            start_idx = jour_selectione * 24
            end_idx = min((jour_selectione + 1) * 24, len(df_input))
            df_filtered = df_input.iloc[start_idx:end_idx].reset_index(drop=True)
            x_axis_title = "Heure du jour"
            graph_title = f"Consommation énergétique - Jour {jour_selectione + 1} (24 heures)"
            
        elif view_option == "Par semaine":
            nb_semaines = max(1, len(df_input) // 168)
            if nb_semaines > 1:
                semaine_selectione = st.slider(
                    "Sélectionner une semaine",
                    min_value=0,
                    max_value=nb_semaines - 1,
                    value=0
                )
            else:
                semaine_selectione = 0
            start_idx = semaine_selectione * 168
            end_idx = min((semaine_selectione + 1) * 168, len(df_input))
            df_filtered = df_input.iloc[start_idx:end_idx].reset_index(drop=True)
            x_axis_title = "Heure de la semaine"
            graph_title = f"Consommation énergétique - Semaine {semaine_selectione + 1} (7 jours)"
            
        else:
            df_filtered = df_input.reset_index(drop=True)
            x_axis_title = "Heure"
            graph_title = "Consommation énergétique - Mois complet"
        
        # Créer le graphique avec les données filtrées
        y_pred_filtered = df_filtered["pred"].values
        x_indices = list(range(len(df_filtered)))
        
        # Recalculer les pics pour les données filtrées
        if y_pred_filtered.size > 1 and (y_pred_filtered.max() - y_pred_filtered.min()) > 0:
            y_pred_range_filt = y_pred_filtered.max() - y_pred_filtered.min()
            peaks_pred_filt, _ = find_peaks(y_pred_filtered, prominence=y_pred_range_filt * 0.1)
        else:
            peaks_pred_filt = np.array([], dtype=int)
        if y_pred_filtered.size > 1:
            if y_pred_filtered[0] > y_pred_filtered[1]: peaks_pred_filt = np.insert(peaks_pred_filt, 0, 0)
            if y_pred_filtered[-1] > y_pred_filtered[-2]: peaks_pred_filt = np.append(peaks_pred_filt, len(y_pred_filtered) - 1)
        
        # Recalculer les pics réels pour les données filtrées
        peaks_real_filt = np.array([], dtype=int)
        if "y_kWh_imported" in df_filtered.columns:
            y_input_filt = df_filtered["y_kWh_imported"].values
            if y_input_filt.size > 1 and (y_input_filt.max() - y_input_filt.min()) > 0:
                y_input_range_filt = y_input_filt.max() - y_input_filt.min()
                peaks_real_filt, _ = find_peaks(y_input_filt, prominence=y_input_range_filt * 0.1)
                if y_input_filt[0] > y_input_filt[1]: peaks_real_filt = np.insert(peaks_real_filt, 0, 0)
                if y_input_filt[-1] > y_input_filt[-2]: peaks_real_filt = np.append(peaks_real_filt, len(y_input_filt)-1)
        
        fig = go.Figure()
        
        # Courbe de prédiction
        fig.add_trace(go.Scatter(
            x=x_indices,
            y=y_pred_filtered, 
            mode="lines+markers", 
            name="Prédiction",
            line=dict(color='#667eea', width=3),
            marker=dict(size=7, opacity=0.7),
            hovertemplate="<b>Index:</b> %{x}<br><b>Consommation:</b> %{y:.2f} kWh<extra></extra>"
        ))
        
        # Pics prédits
        if peaks_pred_filt.size:
            fig.add_trace(go.Scatter(
                x=peaks_pred_filt.tolist(), 
                y=y_pred_filtered[peaks_pred_filt], 
                mode="markers",
                marker=dict(size=12, symbol="triangle-up", color="#ef4444"),
                name="Pics prédits",
                hovertemplate="<b>Pic détecté</b><br>Index: %{x}<br>Consommation: %{y:.2f} kWh<extra></extra>"
            ))
        
        # Données réelles
        if "y_kWh_imported" in df_filtered.columns:
            fig.add_trace(go.Scatter(
                x=x_indices,
                y=df_filtered["y_kWh_imported"].values,
                mode="lines",
                name="Données réelles",
                line=dict(color='#10b981', width=2, dash='dash'),
                hovertemplate="<b>Réel:</b> %{y:.2f} kWh<extra></extra>"
            ))
            
            # Pics réels
            if peaks_real_filt.size:
                fig.add_trace(go.Scatter(
                    x=peaks_real_filt.tolist(), 
                    y=df_filtered["y_kWh_imported"].values[peaks_real_filt], 
                    mode="markers",
                    marker=dict(size=10, symbol="square", color="#10b981"),
                    name="Pics réels",
                    hovertemplate="<b>Pic réel</b><br>Index: %{x}<br>Consommation: %{y:.2f} kWh<extra></extra>"
                ))
        
        fig.update_layout(
            title=dict(
                text=graph_title,
                x=0.5,
                xanchor="center",
                font=dict(size=20, color="#ffffff")
            ),
            xaxis_title=x_axis_title,
            yaxis_title="Consommation (kWh)",
            template="plotly_dark",
            hovermode="x unified",
            height=600,
            margin=dict(l=60, r=60, t=80, b=60),
            plot_bgcolor='rgba(30, 30, 40, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, sans-serif", size=12, color="#e0e0e0"),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(102, 126, 234, 0.1)',
                zeroline=False,
                color='#a0aec0'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(102, 126, 234, 0.1)',
                zeroline=False,
                color='#a0aec0'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(30, 30, 40, 0.7)',
                bordercolor='rgba(102, 126, 234, 0.3)',
                borderwidth=1
            )
        )
        
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------------------------------
        # Statistiques détaillées
        # -------------------------------------------------------
        st.markdown("<div class='section-title'>Statistiques détaillées</div>", unsafe_allow_html=True)
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            # Moyenne par heure
            consommation_moyenne = y_pred.mean()
            st.metric(
                label="Consommation moyenne (par heure)",
                value=f"{consommation_moyenne:.2f} kWh",
                delta=f"Max: {y_pred.max():.2f} kWh"
            )
        
        
        with stat_col2:
            # Consommation maximale (par heure)
            st.metric(
                label="Consommation maximale (par heure)",
                value=f"{y_pred.max():.2f} kWh",
                delta="Charge pic"
            )
        
        with stat_col3:
            if "y_kWh_imported" in df_input.columns:
                mae = np.mean(np.abs(df_input["y_kWh_imported"] - df_input["pred"]))
                st.metric(
                    label="Erreur moyenne (MAE)",
                    value=f"{mae:.2f} kWh",
                    delta="±" + f"{mae:.2f}"
                )

        # -------------------------------------------------------
        # Tableau complet
        # -------------------------------------------------------
        st.markdown("<div class='section-title'>Données complètes avec prédictions</div>", unsafe_allow_html=True)
        st.markdown("<div class='dataframe-wrapper'>", unsafe_allow_html=True)
        st.dataframe(df_input, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # -------------------------------------------------------
        # Téléchargement des résultats
        # -------------------------------------------------------
        st.markdown("<div class='section-title'>Exporter les résultats</div>", unsafe_allow_html=True)
        csv = df_input.to_csv(index=False)
        st.download_button(
            label="Télécharger les résultats en CSV",
            data=csv,
            file_name="predictions_energetiques.csv",
            mime="text/csv"
        )
