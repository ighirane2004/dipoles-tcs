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
        st.subheader("1. Introduction et Classification")
        st.warning("🛑 **ACTION REQUISE :** Allez au bureau du professeur.")
        st.markdown("""
        * Observez le multimètre branché aux bornes d'une pile, puis d'une lampe débranchée.
        * Que vaut la tension $U$ quand le courant $I = 0$ ?
        * Complétez le paragraphe 1 de votre cours (Définition d'un dipôle actif et passif).
        """)
        
    elif st.session_state.etape == 2:
        st.subheader("2. Le Conducteur Ohmique (Rappel)")
        st.info("Sélectionnez 'Conducteur Ohmique' dans le simulateur.")
        st.markdown("""
        * Observez la courbe. Elle passe par l'origine et forme une droite.
        * Jouez avec le curseur de résistance. 
        * Remplissez les trous correspondants dans votre polycopié : le dipôle est **passif** et **linéaire**.
        """)
        
    elif st.session_state.etape == 3:
        st.subheader("3. La Lampe à incandescence")
        st.info("Sélectionnez 'Lampe' dans le simulateur.")
        st.markdown("""
        * Passez la souris sur la courbe pour lire les coordonnées.
        * Notez que l'allure n'est plus une droite.
        * Complétez la leçon : La lampe est un dipôle **passif**, **non linéaire**, mais elle reste **symétrique**.
        """)
        
    elif st.session_state.etape == 4:
        st.subheader("4. La Varistance (VDR)")
        st.info("Sélectionnez 'Varistance (VDR)' dans le simulateur.")
        st.markdown("""
        * Observez comment la tension réagit aux variations de courant.
        * Quel est le rôle de ce composant selon vous ? (Aide : regardez ce qui se passe pour des courants très élevés).
        * Tracez l'allure de la courbe dans le cadre de votre polycopié.
        """)

    elif st.session_state.etape == 5:
        st.subheader("5. La Diode à Jonction")
        st.warning("🛑 **ACTION REQUISE :** Allez au bureau du professeur.")
        st.markdown("""
        * Insérez une vraie diode dans le circuit. Que fait l'ampèremètre si on la branche à l'envers ?
        * **De retour au simulateur** : Sélectionnez 'Diode à jonction'.
        * Notez l'asymétrie totale. Lisez sur l'axe des abscisses la valeur de la tension $U_s$ où le courant décolle.
        * Remplissez le tableau récapitulatif (Sens direct / Sens bloqué).
        """)

    elif st.session_state.etape == 6:
        st.subheader("6. La Diode Zener & LED")
        st.info("Observez successivement la 'Diode Zener' puis la 'LED'.")
        st.markdown("""
        * Comparez la Zener avec la diode simple étudiée précédemment. Que se passe-t-il dans les tensions négatives ?
        * Relevez la valeur de la tension Zener $U_z$ (le claquage réversible).
        * Notez la tension de seuil plus élevée de la LED.
        * Reportez-vous à la remarque de votre cours sur les "caractéristiques idéalisées" pour simplifier ces courbes.
        """)

    elif st.session_state.etape == 7:
        st.subheader("7. La Photorésistance (LDR)")
        st.warning("🛑 **ACTION REQUISE :** Allez manipuler au bureau.")
        st.markdown("""
        * Cachez la LDR avec vos mains pour faire l'obscurité. Que fait l'ohmmètre ?
        * **Sur le simulateur** : Utilisez le curseur de luminosité.
        * Concluez sur la relation entre lumière et résistance dans votre cours.
        """)

    elif st.session_state.etape == 8:
        st.subheader("8. La Thermistance (CTN)")
        st.warning("🛑 **ACTION REQUISE :** Allez manipuler au bureau.")
        st.markdown("""
        * Approchez une source de chaleur de la thermistance. Que fait la résistance ?
        * **Sur le simulateur** : Manipulez le curseur de température pour vérifier le modèle mathématique.
        * Complétez la dernière partie de votre polycopié.
        """)

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.session_state.etape > 1:
            st.button("⬅️ Étape Précédente", on_click=prev_step, use_container_width=True)
    with col_btn2:
        if st.session_state.etape < 8:
            st.button("Étape Suivante ➡️", on_click=next_step, use_container_width=True)
