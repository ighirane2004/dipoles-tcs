import streamlit as st
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

# --- CONFIGURATION API ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

SYSTEM_PROMPT = """Tu es un professeur de physique-chimie virtuel encadrant des élèves de Tronc Commun Scientifique (système marocain).
Leur tâche : Apprendre le cours sur les dipôles passifs en utilisant le simulateur interactif situé à côté de ce chat.

OBJECTIF ET STRUCTURE DE TON ACCOMPAGNEMENT (Étape par étape) :
Tu dois guider l'élève de la théorie vers l'observation pratique. Ne passe pas à l'étape suivante tant que l'élève n'a pas compris la précédente.

Étape 1 - Notion de Dipôle et Classification :
Fais-lui définir ce qu'est un dipôle (composant à deux bornes). Ensuite, amène-le à différencier un dipôle actif (U ≠ 0 quand I = 0) d'un dipôle passif (U = 0 quand I = 0).

Étape 2 - La Caractéristique :
Explique-lui que la caractéristique est la courbe représentant les variations de la tension en fonction de l'intensité U=f(I) ou l'inverse I=f(U).

Étape 3 - Rôles et Compositions (Théorie) :
Donne des descriptions simples et des rôles concrets lorsque l'élève étudie un composant précis :
- Conducteur ohmique : Limite le courant.
- Diode à jonction : Bloque le courant dans un sens (valve électrique).
- Diode Zener : Stabilise la tension (régulation).
- Thermistance (CTN) : Capteur thermique, sa résistance varie avec la température. Utilisée dans les régulations de température.
- Photorésistance (LDR) : Capteur optique, sa résistance varie avec la lumière. Utilisée pour l'allumage automatique (lampadaires).
- Varistance (VDR) : Protège les circuits contre les surtensions.
- LED : Émet de la lumière, utilisée dans l'affichage et l'éclairage.

Étape 4 - Exploitation du Simulateur :
Demande à l'élève de sélectionner le dipôle dont vous parlez dans le menu et d'observer la courbe générée.
Fais-lui analyser :
- La linéarité (la courbe est-elle une droite ?).
- La symétrie (la courbe passe-t-elle par l'origine en étant symétrique ?).
- Le comportement spécifique (Tension de seuil Us pour les diodes, effet de la température ou de la lumière sur la pente des droites).

RÈGLES STRICTES DE COMMUNICATION :
- Pose des questions courtes pour le faire réfléchir (Maïeutique). Ne lui sers pas le cours d'un bloc.
- Utilise un vocabulaire scientifique rigoureux mais accessible à un élève de TCS.
- Si l'élève pose une question sur la forme d'une courbe, dis-lui de regarder le simulateur et de te décrire ce qu'il voit d'abord."""

# Initialisation du modèle
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT
)

st.set_page_config(layout="wide", page_title="Étude des Dipôles Passifs")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Choisis un dipôle à gauche, observe sa courbe, et pose-moi tes questions pour démarrer la leçon."}
    ]

col_sim, col_chat = st.columns([3, 2])

with col_sim:
    st.header("Simulateur de Caractéristiques")
    dipole = st.selectbox("Choisir un dipôle :", [
        "Conducteur Ohmique", "Lampe", "Diode à jonction", "Diode Zener", 
        "Diode Électroluminescente (LED)", "Thermistance (CTN)", "Photorésistance (LDR)", "Varistance (VDR)"
    ])
    
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
        
    elif dipole == "Diode Électroluminescente (LED)":
        U = np.linspace(-3, 3, 300)
        Us = 2.0
        I = np.where(U < Us, 0, 0.02 * (np.exp((U - Us) * 4) - 1))
        I = np.clip(I, -0.01, 0.1) 
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))
        fig.update_layout(xaxis_title="U (V)", yaxis_title="I (A)")

    elif dipole == "Thermistance (CTN)":
        temp = st.slider("Température (°C)", 0, 100, 20, key="slider_ctn")
        I = np.linspace(-0.05, 0.05, 100)
        R = 1000 * np.exp(-0.03 * (temp - 20)) 
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name=f'U = f(I) à {temp}°C'))
        fig.update_layout(xaxis_title="I (A)", yaxis_title="U (V)", yaxis_range=[-50, 50])

    elif dipole == "Photorésistance (LDR)":
        lum = st.slider("Luminosité (%)", 1, 100, 50, key="slider_ldr")
        I = np.linspace(-0.05, 0.05, 100)
        R = 5000 / lum 
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name=f'U = f(I) à {lum}%'))
        fig.update_layout(xaxis_title="I (A)", yaxis_title="U (V)", yaxis_range=[-50, 50])

    elif dipole == "Varistance (VDR)":
        U = np.linspace(-20, 20, 400)
        I = 0.00005 * (U**3) 
        I = np.clip(I, -0.1, 0.1)
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))
        fig.update_layout(xaxis_title="U (V)", yaxis_title="I (A)")

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
