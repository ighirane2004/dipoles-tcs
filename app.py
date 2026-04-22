import streamlit as st
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

# --- CONFIGURATION API ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

SYSTEM_PROMPT = """Tu es un professeur de physique-chimie virtuel accompagnant des élèves de Tronc Commun Scientifique (système marocain).
Leur tâche : Étudier les caractéristiques (U=f(I) ou I=f(U)) des dipôles passifs via le simulateur à l'écran.
RÈGLES STRICTES :
1. NE DONNE JAMAIS LA RÉPONSE DIRECTE. Ton rôle est maïeutique.
2. Si l'élève pose une question directe, renvoie-le TOUJOURS à l'observation de la courbe.
3. Utilise le vocabulaire TCS (linéaire, symétrique, tension de seuil, passant, bloqué).
4. Sois très bref. Une seule question de relance à la fois."""

# Initialisation du modèle
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT
)

st.set_page_config(layout="wide", page_title="Étude des Dipôles Passifs")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Choisis un dipôle à gauche, observe sa courbe, et pose-moi tes questions si tu as besoin d'aide."}
    ]

col_sim, col_chat = st.columns([3, 2])

with col_sim:
    st.header("Simulateur de Caractéristiques")
    dipole = st.selectbox("Choisir un dipôle :", ["Conducteur Ohmique", "Lampe", "Diode à jonction", "Diode Zener", "Thermistance (CTN)"])
    
    fig = go.Figure()
    
    if dipole == "Conducteur Ohmique":
        I = np.linspace(-0.2, 0.2, 200)
        U = 100 * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name='U = f(I)'))
        fig.update_layout(xaxis_title="I (A)", yaxis_title="U (V)")

    elif dipole == "Lampe":
        I = np.linspace(-0.2, 0.2, 200)
        U = 20 * I + 1500 * (I**3)
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name='U = f(I)'))
        fig.update_layout(xaxis_title="I (A)", yaxis_title="U (V)")

    elif dipole == "Diode à jonction":
        U = np.linspace(-2, 1, 300)
        Us = 0.6
        I = np.where(U < Us, 0, 0.05 * (np.exp((U - Us) * 5) - 1))
        I = np.clip(I, -0.01, 0.1) 
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))
        fig.update_layout(xaxis_title="U (V)", yaxis_title="I (A)")

    elif dipole == "Diode Zener":
        U = np.linspace(-8, 2, 500)
        Us = 0.6
        Uz = -6.2
        I = np.zeros_like(U)
        I[U >= Us] = 0.05 * (np.exp((U[U >= Us] - Us) * 5) - 1)
        I[U <= Uz] = -0.05 * (np.exp(-(U[U <= Uz] - Uz) * 5) - 1)
        I = np.clip(I, -0.1, 0.1)
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))
        fig.update_layout(xaxis_title="U (V)", yaxis_title="I (A)")
        
    elif dipole == "Thermistance (CTN)":
        temp = st.slider("Température (°C)", 0, 100, 20, key="slider_ctn")
        I = np.linspace(-0.05, 0.05, 100)
        R = 1000 * np.exp(-0.03 * (temp - 20)) 
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name=f'U = f(I) à {temp}°C'))
        fig.update_layout(xaxis_title="I (A)", yaxis_title="U (V)", yaxis_range=[-50, 50])

    # Application du style sombre
    fig.update_traces(line=dict(width=4, color='#7fbfff'))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='#161b22',
        paper_bgcolor='#161b22',
        font=dict(color="#e6e6e6", size=14, family="Arial"),
        xaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)',
            zeroline=True, zerolinewidth=2, zerolinecolor='rgba(255,255,255,0.3)',
            title_font=dict(size=14, color="#a3a3a3")
        ),
        yaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)',
            zeroline=True, zerolinewidth=2, zerolinecolor='rgba(255,255,255,0.3)',
            title_font=dict(size=14, color="#a3a3a3")
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

with col_chat:
    st.header("Tuteur IA")
    
    # Gestion de l'historique Gemini
    if "gemini_chat" not in st.session_state:
        st.session_state.gemini_chat = model.start_chat(history=[])
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Pose ta question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        try:
            response = st.session_state.gemini_chat.send_message(prompt)
            reponse_ia = response.text
        except Exception as e:
            reponse_ia = f"Erreur technique de l'API : {e}"
            
        st.session_state.messages.append({"role": "assistant", "content": reponse_ia})
        with st.chat_message("assistant"):
            st.markdown(reponse_ia)