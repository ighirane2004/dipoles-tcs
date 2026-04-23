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
1. C'EST TOI QUI DIRIGES. Pose toujours UNE seule question à la fin de ton message pour faire avancer la leçon.
2. Ne donne JAMAIS le mot manquant directement. Fais-le deviner par l'observation socratique.
3. Fais explicitement référence aux paragraphes du cours de l'élève (ex: "Regarde le paragraphe 2.3...").
4. LECTURE GRAPHIQUE OBLIGATOIRE : Pour chaque dipôle étudié, exige de l'élève qu'il lise une valeur précise sur le simulateur (ex: "Quelle est la valeur de U quand I = 0,1 A ?") avant de passer à l'analyse théorique.
5. ILLUSTRATION MULTIMÉDIA : Si l'élève bloque sur un concept abstrait (tension de seuil, effet Zener), fournis-lui un lien direct vers une COURTE vidéo YouTube (moins de 5 minutes) ou une animation interactive.

PLAN DE LA LEÇON À SUIVRE STRICTEMENT :
1. Introduction : Demande à l'élève combien de bornes possède une lampe. Fais-lui déduire le mot "dipôle".
2. Classification (Paragraphe 1) : Ordonne à l'élève d'aller au bureau du professeur pour mesurer la tension aux bornes d'une pile, d'une lampe et d'une diode débranchées. Fais-lui déduire les définitions d'un dipôle passif et actif.
3. Utilité et Définition (Paragraphe 2.1) : Explique que la caractéristique est la "carte d'identité" du composant.
4. Montage (Paragraphe 2.2) : Ordonne-lui d'aller voir le professeur pour observer le montage réel. Explique l'inversion des pôles (valeurs négatives).
5. Rappel (Hors polycopié) : Fais-lui sélectionner "Conducteur Ohmique" pour une lecture graphique et constater la linéarité.
6. Lampe à incandescence (Paragraphe 2.3) : Lecture graphique et déduction des mots : "passif", "non linéaire", "symétrique".
7. Varistance VDR (Paragraphe 2.4) : Même démarche.
8. Diode à jonction (Paragraphe 2.5) : 
   - ÉTAPE A (Le Réel) : AVANT le simulateur, ordonne à l'élève d'aller voir le professeur pour tester le sens direct et le sens bloqué d'une diode réelle dans un circuit.
   - ÉTAPE B (Le Visuel) : Propose-lui une courte vidéo ou animation montrant le comportement des porteurs de charge dans une diode.
   - ÉTAPE C (Le Graphique) : Une fois cela fait, fais-lui utiliser le simulateur pour identifier Us et remplir le tableau (Interrupteur ouvert/fermé).
9. Diode Zener (Paragraphe 2.6) : Comparaison avec la diode à jonction. Lecture de Uz. Effet Zener.
10. Capteurs (Paragraphes 2.7, 2.8, 2.9) : Manipulation des curseurs de température et luminosité.

Ton ton : Pédagogue, direct, sans flatterie."""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT
)

st.set_page_config(layout="wide", page_title="Étude des Dipôles Passifs")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Nous allons construire ton cours ensemble. Pour commencer, pense à une petite lampe ou au moteur d'un jouet. Combien de points de connexion (bornes) possèdent ces composants pour fonctionner ?"}
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
    elif dipole == "Lampe":
        I = np.linspace(-0.2, 0.2, 200)
        U = 20 * I + 1500 * (I**3)
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name='U = f(I)'))
    elif dipole == "Diode à jonction":
        U = np.linspace(-2, 1, 300)
        Us = 0.6
        I = np.where(U < Us, 0, 0.05 * (np.exp((U - Us) * 5) - 1))
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))
    elif dipole == "Diode Zener":
        U = np.linspace(-8, 2, 500)
        Us, Uz = 0.6, -6.2
        I = np.zeros_like(U)
        I[U >= Us] = 0.05 * (np.exp((U[U >= Us] - Us) * 5) - 1)
        I[U <= Uz] = -0.05 * (np.exp(-(U[U <= Uz] - Uz) * 5) - 1)
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))
    elif dipole == "Diode Électroluminescente (LED)":
        U = np.linspace(-3, 3, 300)
        Us = 2.0
        I = np.where(U < Us, 0, 0.02 * (np.exp((U - Us) * 4) - 1))
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))
    elif dipole == "Thermistance (CTN)":
        temp = st.slider("Température (°C)", 0, 100, 20, key="ctn")
        I = np.linspace(-0.05, 0.05, 100)
        R = 1000 * np.exp(-0.03 * (temp - 20)) 
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name='U = f(I)'))
    elif dipole == "Photorésistance (LDR)":
        lum = st.slider("Luminosité (%)", 1, 100, 50, key="ldr")
        I = np.linspace(-0.05, 0.05, 100)
        R = 5000 / lum 
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name='U = f(I)'))
    elif dipole == "Varistance (VDR)":
        U = np.linspace(-20, 20, 400)
        I = 0.00005 * (U**3) 
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))

    fig.update_traces(line=dict(width=4, color='#7fbfff'))
    fig.update_layout(
        template="plotly_dark", plot_bgcolor='#161b22', paper_bgcolor='#161b22',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Tension U (V)" if "f(U)" not in fig.data[0].name else "Intensité I (A)"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Intensité I (A)" if "f(U)" in fig.data[0].name else "Tension U (V)")
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
                
    if prompt := st.chat_input("Réponds au professeur ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            response = st.session_state.gemini_chat.send_message(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Erreur : {e}"})
        st.rerun()
