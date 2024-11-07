pip install gradio
pip install openai-whisper
pip install deepl

pip install --upgrade --no-cache-dir openai-whisper # Upgrade openai-whisper to ensure you have latest version
import deepl
import whisper  # Importing the whisper library

try:
    import gradio as gr
except ModuleNotFoundError:
    !pip install gradio
    import gradio as gr



# Cargar el modelo de Whisper
model = whisper.load_model("base")

# Configurar la clave de API de DeepL
import os
auth_key = os.getenv("DEEPL_API_KEY")
translator = deepl.Translator(auth_key)

# Función principal que procesa el archivo de audio
def transcribe_and_translate(audio_file):
    # Paso 1: Transcripción en español
    result = model.transcribe(audio_file, language="es")
    spanish_text = result["text"]

    # Paso 2: Traducción al inglés con Whisper
    result_translation = model.transcribe(audio_file, task="translate", language="es")
    english_text = result_translation["text"]

    # Paso 3: Traducción al francés con DeepL
    french_translation = translator.translate_text(english_text, target_lang="FR").text

    return spanish_text, english_text, french_translation

# Crear la interfaz de Gradio
# Removing the 'source' argument from gr.Audio
iface = gr.Interface(
    fn=transcribe_and_translate,  # La función a llamar
    inputs=gr.Audio(type="filepath"),  # Subir archivo de audio
    outputs=[
        gr.Textbox(label="Transcripción en español"),
        gr.Textbox(label="Traducción al inglés"),
        gr.Textbox(label="Traducción al francés"),
    ],
    title="Transcriptor y traductor de audio",
    description="Sube un archivo de audio en español para obtener la transcripción y traducciones automáticas en inglés y francés."
)

# Iniciar la aplicación
iface.launch()