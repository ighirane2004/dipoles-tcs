import streamlit as st
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

st.set_page_config(layout="wide", page_title="Étude des Dipôles Passifs")

# --- 1. CONFIGURATION IA ---
# Remplace par ta vraie clé ou utilise st.secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"]) 

SYSTEM_PROMPT = """
Tu es Salma, assistante de travaux pratiques en physique pour des élèves de Tronc Commun.
RÈGLES ABSOLUES :
1. Tes réponses ne doivent JAMAIS dépasser 3 phrases brèves.
2. Si l'élève a juste : valide très brièvement et dis-lui de cliquer sur le bouton "Passer à l'étape suivante" en bas.
3. Si l'élève a faux ou ne sait pas : ne donne JAMAIS la réponse directe. Utilise UNIQUEMENT l'indice fourni dans le contexte.
4. Ne fais jamais de monologue. Sois directe.
"""

model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction=SYSTEM_PROMPT
)

# --- 2. LE PLAN DE VOL (Ton scénario) ---
plan_de_cours = {
    1: {
        "objectif": "Nombre de bornes",
        "question": "Bonjour ! Lors des dernières leçons, vous avez étudié divers composants électriques. Peux-tu me dire combien de bornes possèdent la plupart de ces composants ?",
        "reponse_attendue": "Deux (2).",
        "indice_si_erreur": "Pense au préfixe 'di' que l'on trouve souvent dans ce chapitre. Combien ça fait ?"
    },
    2: {
        "objectif": "Définition dipôle",
        "question": "Exactement. Notre leçon d'aujourd'hui s'attaque à ces composants qu'on appelle dipôles. D'après ce qu'on vient de dire, peux-tu formuler une définition d'un dipôle ?",
        "reponse_attendue": "Un composant électrique qui possède deux bornes.",
        "indice_si_erreur": "Complète simplement ce texte : Un ........ est un composant électrique qui possède ........ bornes."
    },
    3: {
        "objectif": "Présentation physique",
        "question": "Très bien. Je vais te présenter les dipôles de notre étude : conducteur ohmique, lampe, varistance, diode à jonction, diode Zener. Va voir ton professeur pour observer leur forme et leur taille, puis dis-moi quand c'est fait.",
        "reponse_attendue": "C'est fait / J'ai vu le prof.",
        "indice_si_erreur": "Je ne peux pas t'aider ici. Va voir le professeur pour regarder les composants sur le bureau, puis reviens me le dire."
    },
    4: {
        "objectif": "Mesure tension à vide",
        "question": "Parfait. Maintenant on classe ! Va demander à ton professeur de mesurer la tension aux bornes de la pile, puis de la lampe et de la varistance (sans les brancher dans un circuit). Que remarques-tu pour la tension de la pile par rapport aux autres ?",
        "reponse_attendue": "La tension de la pile n'est pas nulle, les autres sont à 0.",
        "indice_si_erreur": "Regarde bien l'écran du voltmètre. La pile affiche-t-elle 0V ou une autre valeur ? Et la lampe ?"
    },
    5: {
        "objectif": "Actif vs Passif",
        "question": "Bien observé. La pile est un dipôle ACTIF (elle fournit l'énergie, U n'est pas nul même sans courant). La lampe et la varistance sont des dipôles PASSIFS (U=0 si I=0). Peux-tu me résumer la définition d'un dipôle passif ?",
        "reponse_attendue": "Sa tension est nulle quand il n'y a pas de courant.",
        "indice_si_erreur": "Rappelle-toi tes mesures : que vaut la tension (U) d'une lampe quand le courant (I) est à 0 ?"
    },
    6: {
        "objectif": "Matériel pour les caractéristiques",
        "question": "On va s'intéresser uniquement aux dipôles passifs. Pour les étudier, on trace leur 'caractéristique' (courbe U en fonction de I). À ton avis, de quels appareils de mesure avons-nous besoin pour relever ces valeurs ?",
        "reponse_attendue": "Un voltmètre et un ampèremètre.",
        "indice_si_erreur": "Comment s'appelle l'appareil qui mesure la tension ? Et celui qui mesure l'intensité du courant ?"
    }
}

MAX_ETAPES = len(plan_de_cours)

# --- 3. INITIALISATION SESSION ---
if "etape" not in st.session_state:
    st.session_state.etape = 1
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": plan_de_cours[1]["question"]}]

def step_suivante():
    st.session_state.etape += 1
    # On purge le chat pour ne pas surcharger l'écran
    st.session_state.messages = [{"role": "assistant", "content": plan_de_cours[st.session_state.etape]["question"]}]

# --- 4. INTERFACE UTILISATEUR ---
col_sim, col_chat = st.columns([3, 2])

# --- COLONNE GAUCHE : SIMULATEUR ---
with col_sim:
    st.header("Simulateur de Caractéristiques")
    dipole = st.selectbox("Choisir le dipôle à afficher :", [
        "Conducteur Ohmique", "Lampe", "Varistance (VDR)", "Diode à jonction", 
        "Diode Zener", "Photorésistance (LDR)", "Thermistance (CTN)"
    ])
    
    fig = go.Figure()
    
    if dipole == "Conducteur Ohmique":
        R = st.slider("Résistance (Ω)", 10, 500, 100)
        I = np.linspace(-0.2, 0.2, 200)
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines'))
        fig.update_layout(yaxis_range=[-100, 100])
    elif dipole == "Lampe":
        I = np.linspace(-0.2, 0.2, 200)
        U = 20 * I + 1500 * (I**3)
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines'))
    elif dipole == "Varistance (VDR)":
        I = np.linspace(-0.1, 0.1, 400)
        U = np.cbrt(I / 0.00005)
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines'))
    elif dipole == "Diode à jonction":
        U = np.linspace(-2, 1, 300)
        Us = 0.6
        I = np.where(U < Us, 0, 0.05 * (np.exp((U - Us) * 5) - 1))
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines'))
        fig.update_layout(yaxis_range=[-0.02, 0.15])
    elif dipole == "Diode Zener":
        U = np.linspace(-8, 2, 500)
        Us, Uz = 0.6, -6.2
        I = np.zeros_like(U)
        I[U >= Us] = 0.05 * (np.exp((U[U >= Us] - Us) * 5) - 1)
        I[U <= Uz] = -0.05 * (np.exp(-(U[U <= Uz] - Uz) * 5) - 1)
        fig.add_trace(go.Scatter(x=U, y=I, mode='lines'))
        fig.update_layout(yaxis_range=[-0.15, 0.15])
    elif dipole == "Photorésistance (LDR)":
        lum = st.slider("Luminosité (%)", 1, 100, 50)
        I = np.linspace(-0.05, 0.05, 100)
        R = 5000 / lum 
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines'))
    elif dipole == "Thermistance (CTN)":
        temp = st.slider("Température (°C)", 0, 100, 20)
        I = np.linspace(-0.05, 0.05, 100)
        R = 1000 * np.exp(-0.03 * (temp - 20)) 
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines'))

    fig.update_traces(line=dict(width=4, color='#7fbfff'))
    fig.update_layout(
        template="plotly_dark", plot_bgcolor='#161b22', paper_bgcolor='#161b22',
        xaxis=dict(title="Courant (A)" if dipole not in ["Diode à jonction", "Diode Zener"] else "Tension (V)"),
        yaxis=dict(title="Tension (V)" if dipole not in ["Diode à jonction", "Diode Zener"] else "Courant (A)")
    )
    st.plotly_chart(fig, use_container_width=True)

# --- COLONNE DROITE : TUTEUR IA (SALMA) ---
with col_chat:
    st.subheader(f"Étape {st.session_state.etape}/{MAX_ETAPES} : {plan_de_cours[st.session_state.etape]['objectif']}")
    st.markdown("---")
    
    # Affichage du chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Input de l'élève
    if prompt := st.chat_input("Réponds à Salma ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        etape_actuelle = plan_de_cours[st.session_state.etape]
        contexte_cache = f"""
        L'élève vient de répondre : "{prompt}"
        ---
        CONSIGNES POUR TOI (SALMA) POUR CETTE ÉTAPE :
        - Objectif : {etape_actuelle['objectif']}
        - Réponse attendue de l'élève : {etape_actuelle['reponse_attendue']}
        - Si la réponse est correcte : valide-la brièvement et dis-lui de cliquer sur 'Passer à l'étape suivante'.
        - Si la réponse est fausse/incomplète : utilise CET indice pour l'orienter : {etape_actuelle['indice_si_erreur']}
        """
        
        with st.chat_message("assistant"):
            with st.spinner("Salma tape..."):
                response = model.generate_content(contexte_cache)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})

    # Bouton de contrôle du prof/élève
    st.markdown("---")
    if st.session_state.etape < MAX_ETAPES:
        st.button("Passer à l'étape suivante ➡️", on_click=step_suivante, use_container_width=True)
    else:
        st.success("🎉 TP Terminé ! Appelez le professeur pour la synthèse.")
