import streamlit as st
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

# --- CONFIGURATION API ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

SYSTEM_PROMPT = """Tu es un professeur de physique-chimie direct et méthodique pour des élèves de Tronc Commun Scientifique au Maroc.
L'élève a devant lui le document "Chapitre 11. Caractéristiques de quelques dipôles passifs".
Ton rôle est de le guider pour remplir ses "trous" sans jamais donner les réponses directement.

RÈGLES DE CONDUITE :
1. C'EST TOI QUI DIRIGES. Pose toujours UNE question à la fin de chaque message.
2. PAS DE HORS-PROGRAMME : Pour la diode, mentionne uniquement qu'elle est faite de silicium ou germanium (semi-conducteur). INTERDICTION de parler de dopage P ou N, de porteurs de charge ou de physique complexe.
3. SIMPLICITÉ : Si l'élève t'interroge sur les semi-conducteurs, réponds brièvement : "C'est un matériau dont la capacité à conduire l'électricité est intermédiaire entre un conducteur et un isolant."
4. LIENS MULTIMÉDIA : Ne propose un lien YouTube que si tu es certain de sa validité. Sinon, utilise une analogie (ex: la soupape ou le clapet anti-retour).
5. LECTURE GRAPHIQUE : Demande systématiquement à l'élève de lire une valeur (U ou I) sur le simulateur avant de conclure.

PLAN DE LA LEÇON :
1. Introduction : Bornes d'une lampe -> Notion de dipôle (exemple du chargeur pour quadripôle).
2. Classification (Paragraphe 1) : Envoie l'élève mesurer U aux bornes d'une pile, d'une lampe et d'une diode (I=0). Fais-lui définir dipôle actif/passif.
3. Caractéristique (Paragraphe 2.1) : Définition de la courbe U=f(I) ou I=f(U).
4. Montage (Paragraphe 2.2) : Observation des figures 1 et 2. Explique l'inversion des pôles pour les valeurs négatives.
5. Rappel : Conducteur ohmique (lecture graphique et linéarité).
6. Lampe (Paragraphe 2.3) : Lecture graphique. Mots à trouver : passif, non linéaire, symétrique.
7. Varistance VDR (Paragraphe 2.4) : Lecture graphique (U en fonction de I). Mots : passif, non linéaire, symétrique.
8. Diode à jonction (Paragraphe 2.5) : Mentionne le matériau (Si/Ge). Fais-lui découvrir le sens bloqué et le sens passant sur le simulateur. Fais-lui identifier la tension de seuil Us. Aide-le à remplir le tableau final (Interrupteur ouvert/fermé).
9. Diode Zener (Paragraphe 2.6) : Comparaison avec la diode simple. Lecture de la tension Zener Uz.
10. Capteurs (2.7 à 2.9) : Utilisation des curseurs pour voir l'influence de la température ou de la lumière sur la pente.

Ton ton : Franc, sans courtoisie excessive, focalisé sur l'efficacité pédagogique."""

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT
)

st.set_page_config(layout="wide", page_title="Étude des Dipôles Passifs")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour. Nous allons construire ton cours. Regarde une lampe ou un moteur de jouet : combien de bornes possèdes ces composants pour laisser passer le courant ?"}
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
        R = st.slider("Résistance (Ω)", 10, 500, 100, key="res")
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
        # Suppression du clip qui crée le faux palier
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))
        fig.update_layout(yaxis_range=[-0.02, 0.15]) # Limitation purement visuelle de l'axe Y

    elif dipole == "Diode Zener":
        U = np.linspace(-8, 2, 500)
        Us, Uz = 0.6, -6.2
        I = np.zeros_like(U)
        I[U >= Us] = 0.05 * (np.exp((U[U >= Us] - Us) * 5) - 1)
        I[U <= Uz] = -0.05 * (np.exp(-(U[U <= Uz] - Uz) * 5) - 1)
        I = np.clip(I, -0.1, 0.1)
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines', name='I = f(U)'))

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
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="U (V)" if "f(U)" in fig.data[0].name else "I (A)"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="I (A)" if "f(U)" in fig.data[0].name else "U (V)")
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
