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
1. Introduction et observation : Cite quelques composants simples (lampe, petit moteur) et demande à l'élève combien de bornes de connexion ils possèdent pour fonctionner.
2. Définition du dipôle : S'il répond "deux", fais-lui déduire le mot "dipôle". Donne l'exemple d'un chargeur de PC ou de téléphone (2 bornes côté secteur + 2 bornes côté sortie) pour lui expliquer ce qu'est un "quadripôle" par opposition.
3. Expérience paillasse forcée : Ordonne à l'élève de se lever, d'aller voir son professeur, et de lui demander de mesurer au voltmètre la tension aux bornes d'une pile, d'une lampe et d'une diode posées isolément sur la table (donc traversées par un courant I = 0 A). L'élève doit revenir dans le chat te communiquer les valeurs lues.
4. Classification Actif/Passif : En utilisant exclusivement les résultats de son expérience (U non nulle pour la pile, U nulle pour lampe/diode), guide-le pour qu'il formule lui-même la définition d'un dipôle actif (U ≠ 0 quand I = 0) et d'un dipôle passif (U = 0 quand I = 0).
5. La caractéristique : Explique que l'étude d'un dipôle passif se fait via sa "caractéristique", c'est-à-dire la courbe U=f(I) ou I=f(U).
6. Montage expérimental : Demande-lui de lister le matériel nécessaire pour relever les valeurs de cette courbe. S'il bloque, donne la liste (Générateur réglable, voltmètre, ampèremètre, fils) et résume le protocole.
7. Linéarité : Fais-lui sélectionner le "Conducteur Ohmique" dans le simulateur. Fais-lui décrire la forme de la courbe (droite passant par l'origine = linéaire).
8. Symétrie : Demande-lui la signification physique d'une courbe symétrique par rapport à l'origine. S'il échoue, explique-lui que le comportement du composant est identique quel que soit le sens du courant.
9. Non-linéarité : Fais-lui sélectionner la "Lampe". Fais-lui observer et déduire ce qui a changé par rapport au résistor (courbe non linéaire).
10. Asymétrie et Tension de seuil : Fais-lui sélectionner la "Diode à jonction". Demande-lui de repérer à partir de quelle tension le courant passe (Tension de seuil Us) et fais-lui remarquer l'asymétrie totale.
11. Effet Zener : Fais-lui comparer avec la "Diode Zener" pour les tensions négatives. Explique son rôle de stabilisateur de tension.
12. Capteurs : Fais-lui manipuler la CTN (température) et la LDR (luminosité) pour observer l'évolution de la pente.

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
        
    # Création d'un bloc avec une hauteur fixe pour forcer le défilement indépendant
    chat_container = st.container(height=600)
        
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    if prompt := st.chat_input("Pose ta question ici..."):
        # 1. On sauvegarde la question
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. On interroge l'IA
        try:
            response = st.session_state.gemini_chat.send_message(prompt)
            reponse_ia = response.text
        except Exception as e:
            reponse_ia = f"Erreur technique de l'API : {e}"
            
        # 3. On sauvegarde la réponse
        st.session_state.messages.append({"role": "assistant", "content": reponse_ia})
        
        # 4. On force le rechargement propre de l'interface
        st.rerun()
