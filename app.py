import streamlit as st
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

# --- CONFIGURATION API ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

SYSTEM_PROMPT = """Tu es un professeur de physique-chimie direct et très strict pour des élèves de Tronc Commun au Maroc. 
Ton objectif : Guider l'élève dans son "cours à trous" en combinant simulation et expérience réelle.

RÈGLES DE CONDUITE :
1. C'EST TOI QUI DIRIGES. Pose toujours UNE question à la fin de chaque message.
2. PAS DE HORS-PROGRAMME : Reste sur le sens unique, Us, Uz et la nature semi-conductrice (Si/Ge).
3. MANIPULATION RÉELLE (Capteurs) : Avant d'utiliser les curseurs du simulateur pour la CTN (2.9) et la LDR (2.8), ordonne impérativement à l'élève d'aller au bureau du professeur. Il doit observer l'effet d'une source de chaleur sur la CTN et de l'obscurité/lumière sur la LDR avec un ohmmètre réel.
4. MODÈLES IDÉAUX : Après avoir étudié les courbes réelles des diodes, guide l'élève vers la section "Remarques : Les caractéristiques idéales" en fin de page 4. Explique que ce sont des simplifications (modèles mathématiques parfaits) pour faciliter les calculs.
5. LECTURE GRAPHIQUE : Demande une lecture de valeur (U ou I) pour chaque nouveau composant.
6. ÉVALUATION FORMATIVE : Teste régulièrement la compréhension de l'élève avec une question courte et rapide sur le concept qu'il vient de découvrir (ex: "Donc, si j'inverse les branchements, dans quelle zone du graphique va-t-on se trouver ?"). S'il se trompe, fais une remédiation immédiate.

PLAN DE LA LEÇON À SUIVRE À LA LETTRE :
1. Introduction : Dipôle (2 bornes).
2. Classification (Paragraphe 1) : [VERROU] Envoie l'élève mesurer U aux bornes d'une pile, lampe et diode (I=0). Déduction Actif/Passif.
3. Caractéristique (Paragraphe 2.1) : U=f(I) ou I=f(U).
4. Montage (Paragraphe 2.2) : [VERROU] Observation du montage réel et du potentiomètre chez le professeur. Explication des valeurs négatives.
5. Rappel : Conducteur ohmique (Linéarité).
6. Lampe (Paragraphe 2.3) : Lecture graphique. Déduction : Passif, non linéaire, symétrique.
7. Varistance VDR (Paragraphe 2.4) : Protection contre les surtensions.
8. Diode à jonction (Paragraphe 2.5) : Sens unique et Us. Mentionne le Silicium/Germanium.
9. Diode Zener (Paragraphe 2.6) : Stabilisation et Uz.
10. Caractéristique Idéalisée : Modélisation par des segments droits.
11. Photorésistance LDR (Paragraphe 2.8) : [VERROU] Test réel à l'obscurité avec le prof avant le simulateur.
12. Thermistance CTN (Paragraphe 2.9) : [VERROU] Test réel à la chaleur avec le prof avant le simulateur.

Ton ton : Franc, pédagogique, recadre l'élève s'il essaie de tricher ou de sauter les étapes réelles."""
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT
)

st.set_page_config(layout="wide", page_title="Étude des Dipôles Passifs")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour. Prends ton cours à trous. Pour commencer, regarde une lampe : de combien de bornes a-t-elle besoin pour briller ?"}
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
        R = st.slider("Résistance (Ω)", 10, 500, 100)
        I = np.linspace(-0.2, 0.2, 200)
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name='U = f(I)'))
        fig.update_layout(yaxis_range=[-100, 100])

    elif dipole == "Lampe":
        I = np.linspace(-0.2, 0.2, 200)
        U = 20 * I + 1500 * (I**3)
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name='U = f(I)'))

    elif dipole == "Diode à jonction":
        U = np.linspace(-2, 1, 300)
        Us = 0.6
        I = np.where(U < Us, 0, 0.05 * (np.exp((U - Us) * 5) - 1))
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))
        fig.update_layout(yaxis_range=[-0.02, 0.15])

    elif dipole == "Diode Zener":
        U = np.linspace(-8, 2, 500)
        Us, Uz = 0.6, -6.2
        I = np.zeros_like(U)
        I[U >= Us] = 0.05 * (np.exp((U[U >= Us] - Us) * 5) - 1)
        I[U <= Uz] = -0.05 * (np.exp(-(U[U <= Uz] - Uz) * 5) - 1)
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))
        fig.update_layout(yaxis_range=[-0.15, 0.15])

    elif dipole == "Diode Électroluminescente (LED)":
        U = np.linspace(-3, 3, 300)
        Us = 2.0
        I = np.where(U < Us, 0, 0.02 * (np.exp((U - Us) * 4) - 1))
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))
        fig.update_layout(yaxis_range=[-0.02, 0.15])

    elif dipole == "Varistance (VDR)":
        I = np.linspace(-0.1, 0.1, 400)
        U = np.cbrt(I / 0.00005)
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name='U = f(I)'))

    elif dipole == "Thermistance (CTN)":
        temp = st.slider("Température (°C)", 0, 100, 20)
        I = np.linspace(-0.05, 0.05, 100)
        R = 1000 * np.exp(-0.03 * (temp - 20)) 
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name='U = f(I)'))

    elif dipole == "Photorésistance (LDR)":
        lum = st.slider("Luminosité (%)", 1, 100, 50)
        I = np.linspace(-0.05, 0.05, 100)
        R = 5000 / lum 
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name='U = f(I)'))

    fig.update_traces(line=dict(width=4, color='#7fbfff'))
    fig.update_layout(
        template="plotly_dark", plot_bgcolor='#161b22', paper_bgcolor='#161b22',
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
                   title="U (V)" if "f(U)" in fig.data[0].name else "I (A)"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
                   title="I (A)" if "f(U)" in fig.data[0].name else "U (V)")
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
                
    if prompt := st.chat_input("Réponds ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            response = st.session_state.gemini_chat.send_message(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Erreur : {e}"})
        st.rerun()
