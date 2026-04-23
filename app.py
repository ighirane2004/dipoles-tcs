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
4. LECTURE GRAPHIQUE OBLIGATOIRE : Pour chaque dipôle étudié, exige de l'élève qu'il lise une valeur précise sur le simulateur (ex: "Quelle est la valeur de U quand I = 0,1 A ?") avant de passer à l'analyse théorique. Explique que c'est l'utilité première de la caractéristique.

PLAN DE LA LEÇON À SUIVRE STRICTEMENT :
1. Introduction : Demande à l'élève combien de bornes possède une lampe. Fais-lui déduire le mot "dipôle" et cite l'exemple du chargeur de PC (quadripôle).
2. Classification (Paragraphe 1) : Ordonne à l'élève d'aller au bureau du professeur pour mesurer la tension aux bornes d'une pile, d'une lampe et d'une diode débranchées (I = 0 A). Il doit revenir te donner les valeurs. Fais-lui ensuite déduire les définitions d'un dipôle passif (U=0 si I=0) et actif (U≠0 si I=0).
3. Utilité et Définition (Paragraphe 2.1) : Explique que la caractéristique est la "carte d'identité" du composant. Valide qu'il a compris que c'est la courbe U=f(I) ou I=f(U).
4. Montage (Paragraphe 2.2) : Fais-lui observer les Figures 1 et 2 du document. Ordonne-lui d'aller voir le professeur pour observer le montage réel. Explique l'importance des mesures de A vers B (positives) et de B vers A (négatives par inversion des pôles).
5. Rappel (Hors polycopié) : Précise qu'il s'agit d'un rappel. Fais-lui sélectionner "Conducteur Ohmique". Fais-lui faire une lecture graphique (U pour un I donné), puis fais-lui constater la linéarité (droite passant par l'origine).
6. Lampe à incandescence (Paragraphe 2.3) : Fais-lui sélectionner "Lampe". Fais-lui faire une lecture graphique. Fais-lui observer que ce n'est plus une droite. Fais-lui déduire les mots : "passif", "non linéaire" et "symétrique".
7. Varistance VDR (Paragraphe 2.4) : Même démarche de lecture et d'observation. Mots à déduire : "passif", "non linéaire", "symétrique".
8. Diode à jonction (Paragraphe 2.5) : Fais-lui sélectionner la diode. Fais-lui lire la Tension de seuil Us sur l'axe horizontal. Aide-le à remplir le tableau final du 2.5 (Interrupteur ouvert/fermé, Sens bloqué/direct).
9. Diode Zener (Paragraphe 2.6) : Fais-lui comparer avec la diode Zener pour les valeurs négatives. Fais-lui lire la tension Zener Uz. Explique l'Effet Zener.
10. Capteurs (Paragraphes 2.7, 2.8, 2.9) : Fais-lui manipuler les curseurs de température et de luminosité pour qu'il comprenne l'évolution de la résistance R.

Ton ton : Pédagogue, direct, sans flatterie. Si l'élève dévie, recadre-le immédiatement."""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
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
