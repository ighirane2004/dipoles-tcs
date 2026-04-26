import streamlit as st
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

# --- CONFIGURATION IA ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(layout="wide", page_title="Étude des Dipôles Passifs")

# --- GESTION DE LA PROGRESSION ET DE L'IA ---
if "etape" not in st.session_state:
    st.session_state.etape = 1
if "ai_responses" not in st.session_state:
    st.session_state.ai_responses = {}

def clear_ai():
    st.session_state.ai_responses = {}

def next_step():
    st.session_state.etape += 1
    clear_ai()

def prev_step():
    st.session_state.etape -= 1
    clear_ai()

def ask_ai(hidden_prompt, key):
    # Appel API caché avec un prompt lourdement structuré
    response = model.generate_content(hidden_prompt)
    st.session_state.ai_responses[key] = response.text
# --- INTERFACE ---
col_sim, col_guide = st.columns([3, 2])

# --- COLONNE GAUCHE : LE SIMULATEUR ---
with col_sim:
    st.header("Simulateur de Caractéristiques")
    dipole = st.selectbox("Choisir le dipôle à afficher :", [
        "Conducteur Ohmique", "Lampe", "Varistance (VDR)", "Diode à jonction", 
        "Diode Zener", "Diode Électroluminescente (LED)", "Photorésistance (LDR)", "Thermistance (CTN)"
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

    elif dipole == "Varistance (VDR)":
        I = np.linspace(-0.1, 0.1, 400)
        U = np.cbrt(I / 0.00005)
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

    elif dipole == "Photorésistance (LDR)":
        lum = st.slider("Luminosité (%)", 1, 100, 50)
        I = np.linspace(-0.05, 0.05, 100)
        R = 5000 / lum 
        U = R * I
        fig.add_trace(go.Scatter(x=I, y=U, mode='lines', name='U = f(I)'))

    elif dipole == "Thermistance (CTN)":
        temp = st.slider("Température (°C)", 0, 100, 20)
        I = np.linspace(-0.05, 0.05, 100)
        R = 1000 * np.exp(-0.03 * (temp - 20)) 
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

# --- COLONNE DROITE : LE SÉQUENCEUR PÉDAGOGIQUE ---
with col_guide:
    st.header(f"Guide Pratique - Étape {st.session_state.etape}/8")
    st.markdown("---")
    
    if st.session_state.etape == 1:
        st.subheader("1. Fondamentaux : Dipôle Actif vs Passif")
        st.markdown("""
        **Leçon :**
        Un dipôle est **actif** s'il possède une tension à ses bornes même lorsqu'il est débranché (I=0). 
        Un dipôle est **passif** si sa tension est nulle en l'absence de courant.
        
        🛑 **Manipulation :** Allez au bureau mesurer la tension d'une pile puis d'une lampe isolées.
        """)
        if st.button("🤖 IA : Donne-moi un moyen mnémotechnique pour retenir la différence", use_container_width=True):
            prompt = "Agis comme un prof de physique. Donne un moyen mnémotechnique ou une analogie très simple (2 phrases max) pour qu'un élève de 15 ans différencie un dipôle actif (qui donne l'énergie) d'un dipôle passif (qui la reçoit/consomme)."
            with st.spinner("Génération..."): ask_ai(prompt, "actif_passif")
        if "actif_passif" in st.session_state.ai_responses: st.success(st.session_state.ai_responses["actif_passif"])

    elif st.session_state.etape == 2:
        st.subheader("2. Le Conducteur Ohmique")
        st.info("Sélectionnez 'Conducteur Ohmique' dans le simulateur à gauche.")
        st.markdown("""
        **Leçon :**
        Sa caractéristique est une droite qui passe par l'origine. 
        Puisqu'elle passe par zéro, c'est un dipôle **passif**. Puisque c'est une droite, il est **linéaire** (il obéit à la loi d'Ohm : U = R × I).
        """)
        if st.button("🤖 IA : Explique ce que signifie physiquement 'être linéaire'", use_container_width=True):
            prompt = "Explique à un élève de 15 ans ce que signifie le mot 'linéaire' en physique pour un conducteur ohmique (proportionnalité entre U et I). Fais court, 3 phrases max."
            with st.spinner("Génération..."): ask_ai(prompt, "lineaire")
        if "lineaire" in st.session_state.ai_responses: st.success(st.session_state.ai_responses["lineaire"])
        
    elif st.session_state.etape == 3:
        st.subheader("3. La Lampe à incandescence")
        st.info("Sélectionnez 'Lampe' dans le simulateur.")
        st.markdown("""
        **Leçon :**
        La courbe passe par l'origine (passif) et est symétrique. Mais ce n'est plus une droite ! 
        La lampe est un dipôle **non linéaire**. Sa résistance n'est pas constante : elle augmente avec l'intensité à cause de l'échauffement du filament.
        """)
        if st.button("🤖 IA : Pourquoi le filament qui chauffe modifie la courbe ?", use_container_width=True):
            prompt = "Explique simplement pourquoi la résistance d'un filament d'ampoule augmente quand il chauffe (agitation thermique qui freine les électrons). 3 phrases max."
            with st.spinner("Génération..."): ask_ai(prompt, "lampe_chauffe")
        if "lampe_chauffe" in st.session_state.ai_responses: st.success(st.session_state.ai_responses["lampe_chauffe"])

    elif st.session_state.etape == 4:
        st.subheader("4. La Varistance (VDR)")
        st.info("Sélectionnez 'Varistance (VDR)'.")
        st.markdown("""
        **Leçon :**
        La VDR (Voltage Dependent Resistor) a une résistance qui s'effondre quand la tension devient trop forte. 
        Elle agit comme un bouclier : elle est placée en dérivation pour absorber les surtensions brutales et protéger le reste du circuit.
        """)
        if st.button("🤖 IA : Donne une analogie pour comprendre le rôle de protection de la VDR", use_container_width=True):
            prompt = "Donne une analogie (ex: un canal de dérivation pour une rivière en crue) pour expliquer le rôle de protection contre les surtensions d'une varistance VDR. 2 phrases max."
            with st.spinner("Génération..."): ask_ai(prompt, "vdr_analogie")
        if "vdr_analogie" in st.session_state.ai_responses: st.success(st.session_state.ai_responses["vdr_analogie"])

    elif st.session_state.etape == 5:
        st.subheader("5. La Diode à Jonction")
        st.info("Sélectionnez 'Diode à jonction'.")
        st.markdown("""
        **Leçon :**
        C'est un dipôle **polarisé** (asymétrique). 
        - Sens bloqué (U < 0) : Le courant ne passe pas (I = 0).
        - Sens direct : Le courant ne passe que si la tension dépasse une valeur minimale appelée **tension de seuil ($U_s$)**.
        """)
        if st.button("🤖 IA : C'est quoi la 'tension de seuil' ?", use_container_width=True):
            prompt = "Explique le concept de 'tension de seuil' (Us) d'une diode (comme une porte qui nécessite une certaine force pour s'ouvrir). 2 phrases max."
            with st.spinner("Génération..."): ask_ai(prompt, "diode_seuil")
        if "diode_seuil" in st.session_state.ai_responses: st.success(st.session_state.ai_responses["diode_seuil"])

    elif st.session_state.etape == 6:
        st.subheader("6. La Diode Zener")
        st.info("Sélectionnez 'Diode Zener'.")
        st.markdown("""
        **Leçon :**
        Elle se comporte comme une diode normale dans le sens direct. Mais dans le sens inverse, au lieu de rester bloquée indéfiniment, elle laisse passer le courant à partir d'une tension précise : **la tension Zener ($U_z$)**. 
        C'est l'effet d'avalanche, utilisé pour stabiliser une tension.
        """)
        if st.button("🤖 IA : Explique le claquage réversible (Effet Zener)", use_container_width=True):
            prompt = "Explique l'effet d'avalanche (claquage réversible) de la diode Zener avec une analogie avec l'eau (soupape de sécurité). 3 phrases max."
            with st.spinner("Génération..."): ask_ai(prompt, "zener_avalanche")
        if "zener_avalanche" in st.session_state.ai_responses: st.success(st.session_state.ai_responses["zener_avalanche"])

    elif st.session_state.etape == 7:
        st.subheader("7. La Photorésistance (LDR)")
        st.info("Sélectionnez 'Photorésistance (LDR)'. Jouez avec la luminosité.")
        st.markdown("""
        **Leçon :**
        La Light Dependent Resistor est un capteur. 
        Dans l'obscurité totale, sa résistance est énorme (elle se comporte presque comme un isolant). Plus elle est éclairée, plus sa résistance diminue et laisse passer le courant.
        """)
        if st.button("🤖 IA : Donne des exemples d'utilisation de la LDR au quotidien", use_container_width=True):
            prompt = "Cite 2 exemples d'utilisation courante d'une photorésistance (LDR) dans la vie de tous les jours (ex: lampadaires automatiques). Fais court."
            with st.spinner("Génération..."): ask_ai(prompt, "ldr_exemples")
        if "ldr_exemples" in st.session_state.ai_responses: st.success(st.session_state.ai_responses["ldr_exemples"])

    elif st.session_state.etape == 8:
        st.subheader("8. La Thermistance (CTN)")
        st.info("Sélectionnez 'Thermistance (CTN)'. Jouez avec la température.")
        st.markdown("""
        **Leçon :**
        La CTN (Coefficient de Température Négatif) voit sa résistance **diminuer** quand la température **augmente**. 
        C'est le composant principal des thermomètres électroniques et des sécurités anti-surchauffe.
        """)
        if st.button("🤖 IA : Quelle est la différence entre une CTN et une CTP ?", use_container_width=True):
            prompt = "Explique brièvement et simplement la différence entre une thermistance CTN et une thermistance CTP vis-à-vis de la température."
            with st.spinner("Génération..."): ask_ai(prompt, "ctn_ctp")
        if "ctn_ctp" in st.session_state.ai_responses: st.success(st.session_state.ai_responses["ctn_ctp"])

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.session_state.etape > 1:
            st.button("⬅️ Étape Précédente", on_click=prev_step, use_container_width=True)
    with col_btn2:
        if st.session_state.etape < 8:
            st.button("Étape Suivante ➡️", on_click=next_step, use_container_width=True)
