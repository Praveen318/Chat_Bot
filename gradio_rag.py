import PIL.Image
import gradio as gr
import base64
import os
from deep_translator import GoogleTranslator
import google.generativeai as genai

# Set Google API key securely
os.environ['GOOGLE_API_KEY'] = "AIzaSyDwIEycJiORGYfDdzFpsE6VrtM_F8bmtLw"  # Replace with your actual API key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Create the Models
txt_model = genai.GenerativeModel('gemini-pro')
vis_model = genai.GenerativeModel('gemini-pro-vision')

def image_to_base64(image_path):
    try:
        with open(image_path, 'rb') as img:
            encoded_string = base64.b64encode(img.read())
        return encoded_string.decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def add_query_to_history(history, txt, img_path, pdf_path):
    if not img_path and not pdf_path:
        history.append((txt, None))
    elif img_path:
        base64_str = image_to_base64(img_path)
        if base64_str:
            data_url = f"data:image/jpeg;base64,{base64_str}"
            history.append((f"{txt} ![]({data_url})", None))
        else:
            history.append((txt, "Error processing image"))
    elif pdf_path:
        history.append((f"{txt} (PDF uploaded: {pdf_path})", None))
    return history

def generate_llm_response(history, text, img_path, pdf_path, target_language):
    try:
        if not img_path and not pdf_path:
            response = txt_model.generate_content(text)
            translated_text = translate_text(response.text, target_language)
            history.append((None, translated_text))
        elif img_path:
            img = PIL.Image.open(img_path)
            response = vis_model.generate_content([text, img])
            translated_text = translate_text(response.text, target_language)
            history.append((None, translated_text))
        elif pdf_path:
            response = txt_model.generate_content(f"{text} (PDF content processing not implemented)")
            translated_text = translate_text(response.text, target_language)
            history.append((None, translated_text))
    except Exception as e:
        history.append((None, f"Error generating response: {e}"))
    return history

def translate_text(text, target_language):
    if target_language == "en":
        return text
    try:
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        print(f"Error translating text: {e}")
        return text

# Gradio Interface
with gr.Blocks() as app:
    with gr.Row():
        image_box = gr.Image(type="filepath", label="Upload Image")
        pdf_box = gr.File(type="filepath", label="Upload PDF")
        chatbot = gr.Chatbot(scale=2, height=750)

    text_box = gr.Textbox(placeholder="Enter text and press enter, or upload an image/PDF", container=True)
    language_dropdown = gr.Dropdown(label="Select Language", choices=["en", "es", "fr", "de", "zh"], value="en")
    submit_button = gr.Button("Submit")

    submit_button.click(
        fn=add_query_to_history, 
        inputs=[chatbot, text_box, image_box, pdf_box], 
        outputs=chatbot
    ).then(
        fn=generate_llm_response, 
        inputs=[chatbot, text_box, image_box, pdf_box, language_dropdown], 
        outputs=chatbot
    )

app.queue()
app.launch(debug=True)
