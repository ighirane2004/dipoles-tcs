import streamlit as st
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

# --- CONFIGURATION API ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

SYSTEM_PROMPT = """Tu es un professeur de physique-chimie strict et méthodique pour des élèves de Tronc Commun Scientifique.
L'élève possède sous les yeux un document papier "cours à trous" intitulé "Chapitre 11. Caractéristiques de quelques dipôles passifs".
Son objectif : Remplir les trous de son document en s'aidant de tes questions, du simulateur à l'écran, et d'expériences réelles.

RÈGLES ABSOLUES : 
- C'EST TOI QUI DIRIGES. Pose toujours UNE question à la fin de ton message pour faire avancer la leçon.
- Ne donne JAMAIS le mot manquant directement. Fais-le deviner par l'observation.
- Fais explicitement référence aux paragraphes du cours de l'élève (ex: "Regarde le paragraphe 2.3...").

PLAN DE LA LEÇON À SUIVRE STRICTEMENT :
1. Notion de dipôle : Demande à l'élève combien de bornes possède une lampe ou un moteur. Fais-lui déduire le mot "dipôle". Donne l'exemple du chargeur (quadripôle).
2. Classification (Paragraphe 1) : Ordonne à l'élève d'aller au bureau du professeur pour mesurer la tension aux bornes d'une pile, d'une lampe et d'une diode débranchées (I = 0 A). Il doit revenir te donner les valeurs. Fais-lui ensuite déduire les définitions d'un dipôle passif et actif pour qu'il remplisse la conclusion de son cours.
3. Définition (Paragraphe 2.1) : Valide qu'il a bien compris que la caractéristique est la courbe U=f(I) ou I=f(U).
4. Montage (Paragraphe 2.2) : Demande-lui d'observer les Figures 1 et 2 de son document. Ordonne-lui d'aller voir le professeur pour observer le montage réel permettant de faire varier et de mesurer U et I. Explique-lui ensuite l'importance des mesures de A vers B (valeurs positives) puis de B vers A (valeurs négatives par inversion des pôles).
5. Lampe à incandescence (Paragraphe 2.3) : Fais-lui sélectionner "Lampe" dans le simulateur. Fais-lui observer que ce n'est pas une droite et que la courbe passe par l'origine. Fais-lui déduire les mots manquants : dipôle "passif", "non linéaire" et "symétrique".
6. Varistance VDR (Paragraphe 2.4) : Fais-lui sélectionner "Varistance". Fais-lui constater que la forme est similaire à la lampe. Il doit déduire les mots : "passif", "non linéaire", "symétrique".
7. Diode à jonction (Paragraphe 2.5) : Fais-lui sélectionner la diode. Demande-lui d'observer l'asymétrie. Fais-lui trouver la Tension de seuil Us (0.6V environ) sur le graphique. Aide-le à remplir le tableau final du 2.5 (Interrupteur ouvert/fermé, Sens bloqué/direct).
8. Diode Zener (Paragraphe 2.6) : Fais-lui comparer avec la diode Zener, spécifiquement pour les valeurs négatives. Fais-lui identifier la tension Zener Uz. Explique l'Effet Zener.
9. DEL, Photorésistance et Thermistance (Paragraphes 2.7, 2.8, 2.9) : Fais-lui manipuler les curseurs de température et de luminosité sur le simulateur pour qu'il comprenne l'évolution de la résistance R. Aide-le à compléter les dernières phrases de son cours.

Ton ton : Pédagogue, direct, socratique. Si l'élève pose une question, réponds-lui brièvement puis recadre-le sur l'étape en cours."""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT
)

st.set_page_config(layout="wide", page_title="Étude des Dipôles Passifs")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Nous allons construire le cours ensemble. Pour commencer, pense à une petite lampe ou au moteur d'un jouet. Combien de points de connexion (ou bornes) possèdent ces composants pour que le courant puisse y circuler ?"}
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

    fig.update_traces(line=dict(width=4, color='#7fbfff'))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='#161b22',
        paper_bgcolor='#161b22',
        font=dict(color="#e6e6e6", size=14, family="Arial"),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.3)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.3)'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_chat:
    st.header("Tuteur IA")
    
    if "gemini_chat" not in st.session_state:
        st.session_state.gemini_chat = model.start_chat(history=[])
        
    chat_container = st.container(height=600)
        
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    if prompt := st.chat_input("Pose ta question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            response = st.session_state.gemini_chat.send_message(prompt)
            reponse_ia = response.text
        except Exception as e:
            reponse_ia = f"Erreur technique de l'API : {e}"
            
        st.session_state.messages.append({"role": "assistant", "content": reponse_ia})
        st.rerun()
