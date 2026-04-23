import streamlit as st
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

# Configuration API sécurisée
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

SYSTEM_PROMPT = """Tu es un professeur de physique-chimie strict et méthodique pour des élèves de Tronc Commun Scientifique (système marocain).
Leur tâche : Découvrir la notion de dipôle passif et analyser leurs caractéristiques via le simulateur interactif à l'écran.

RÈGLE ABSOLUE : C'EST TOI QUI DIRIGES LA LEÇON. Ne réponds jamais passivement. À chaque fin de message, pose UNE question précise à l'élève pour le faire avancer à l'étape suivante. Ne donne JAMAIS les définitions ou les conclusions directement, fais-les deviner par l'observation.

PLAN DE LA LEÇON À SUIVRE STRICTEMENT :
1. Notion de dipôle : Demande à l'élève ce qu'est un dipôle et exige des exemples concrets. Valide ou corrige ses propositions.
2. Classification : Demande-lui la différence entre un dipôle actif et passif. Amène-le à la conclusion stricte : pour un dipôle passif, U = 0 V quand I = 0 A.
3. La caractéristique : Explique clairement que l'étude d'un dipôle se fait via le tracé de sa "caractéristique", c'est-à-dire la courbe U=f(I) ou I=f(U).
4. Montage expérimental : Demande-lui quel matériel est nécessaire pour relever les valeurs de cette courbe. S'il bloque, liste le matériel (Générateur réglable, voltmètre, ampèremètre, fils) et résume brièvement le protocole de mesure.
5. Linéarité : Fais-lui sélectionner le "Conducteur Ohmique" dans le simulateur. Fais-lui décrire la forme de la courbe (droite passant par l'origine = linéaire).
6. Symétrie : Demande-lui la signification physique d'une courbe symétrique. S'il échoue, explique-lui que cela signifie que le comportement du composant est identique quel que soit le sens du branchement.
7. Non-linéarité : Fais-lui sélectionner la "Lampe". Fais-lui observer et déduire ce qui a changé par rapport au résistor (courbe non linéaire).
8. Asymétrie et Tension de seuil : Fais-lui sélectionner la "Diode à jonction". Demande-lui de repérer à partir de quelle tension le courant passe (Tension de seuil Us) et fais-lui remarquer l'asymétrie totale.
9. Effet Zener : Fais-lui comparer avec la "Diode Zener" pour les tensions négatives. Explique son rôle de stabilisateur de tension.
10. Capteurs : Fais-lui manipuler la CTN (curseur de température) et la LDR (curseur de luminosité) pour observer l'évolution de la pente.
11. Action physique : Lorsque c'est pertinent au cours de l'échange, ordonne à l'élève d'appeler son professeur pour observer sur la paillasse le sens direct d'une vraie diode PN ou pour mesurer la résistance d'une LDR réelle sous différents éclairages.
Ton ton : Pédagogue, direct, socratique."""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT
)

st.set_page_config(layout="wide", page_title="Étude des Dipôles Passifs")

# Amorce proactive du chatbot
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Nous allons étudier les dipôles ensemble. Pour commencer, sais-tu faire la différence entre un dipôle passif et un dipôle actif vis-à-vis du courant et de la tension ?"}
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

    # Thème visuel
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
