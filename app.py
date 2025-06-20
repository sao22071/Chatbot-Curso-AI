import streamlit as st 
import utils 

st.set_page_config(page_title="ChatBot Básico",
                                page_icon="❤",
                                layout="wide")

st.title("Chatbot Básico Curso AI")

#Historial
#session_state -> comportamiento de la memoria cache mientras se trabaja 
if "history" not in st.session_state:
  st.session_state.history = []

#contexto 
if "context" not in st.session_state:
  st.session_state.context = []

#Construimoos el espacio., emisor-mensaje
# a medida que la persona escriba se cambia el estado, y este hace que el bot genere una respuesta
#sender es la persona que envia y el mensaje es msg
for sender, msg in st.session_state.history:
  if sender == "Tú":
    st.markdown(f'**🦄💭{sender}:**{msg}')
  else:
    st.markdown(f'**🤖{sender}:**{msg}')


#sino hay entrada 
# se declara la entrada de usuario
if "user_input" not in st.session_state:
  st.session_state.user_input = ""

#Procesamiento de la entrada 
def send_msg():
  user_input = st.session_state.user_input.strip()
  if user_input: #si hay valor se procesa la información que ingrese el usuario
    tag = utils.predict_class(user_input)
    st.session_state.context.append(tag)
    response = utils.get_response(tag,st.session_state.context)
    st.session_state.history.append(('Tú', user_input))
    st.session_state.history.append(('Bot', response))
    st.session_state.user_input = ''

#creamos el campo de texto porque en streamlit aparece todo en orden como uno lo ponga 
st.text_input("Escribe tu mensaje:", key="user_input", on_change=send_msg)
















