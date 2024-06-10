import gradio as gr
import PIL.Image
import base64
import os
from deep_translator import GoogleTranslator
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from upload_docs import upload_loader_func
from response import articulation_messages_func
from tempfile import mkdtemp
from webloader import web_upload_loader_func
import evaluate
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
from response_duck import articulation_messages_duck_func
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from io import BytesIO
from diffusers import StableDiffusionPipeline
import pymongo


# Set Google API key securely
os.environ['GOOGLE_API_KEY'] = "AIzaSyDwIEycJiORGYfDdzFpsE6VrtM_F8bmtLw"  # Replace with your actual API key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Connect to MongoDB
MONGO_URI = "mongodb+srv://Praveen0309:Praveen0309@cluster1.hidzlai.mongodb.net/"
client = pymongo.MongoClient(MONGO_URI)
db = client["Try3"]
collection = db["Col1"]

# Create the Models
txt_model = genai.GenerativeModel('gemini-pro')
vis_model = genai.GenerativeModel('gemini-pro-vision')

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model_id = "HuggingFaceH4/vsft-llava-1.5-7b-hf-trl"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
base_model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, torch_dtype=torch.float16)

# Load the PEFT Lora adapter
peft_lora_adapter_path = "Praveen0309/llava-1.5-7b-hf-ft-mix-vsft-3"
peft_lora_adapter = PeftModel.from_pretrained(base_model, peft_lora_adapter_path, adapter_name="lora_adapter")
base_model.load_adapter(peft_lora_adapter_path, adapter_name="lora_adapter")

processor = AutoProcessor.from_pretrained("HuggingFaceH4/vsft-llava-1.5-7b-hf-trl")

model_id_gen = "OFA-Sys/small-stable-diffusion-v0"
pipe = StableDiffusionPipeline.from_pretrained(model_id_gen, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def store_history_in_mongo(history_entry):
    try:
        collection.insert_one(history_entry)
    except Exception as e:
        print(f"Error storing history in MongoDB: {e}")



def image_to_base64(image_path):
    try:
        with open(image_path, 'rb') as img:
            encoded_string = base64.b64encode(img.read())
        return encoded_string.decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def image_to_base64_1(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    return base64.b64encode(buffered.getvalue()).decode()

# def add_query_to_history(history, txt, img_path, pdf_path,url_path):
#     if not img_path and not pdf_path and not url_path:
#         history.append((txt, None))
#     elif img_path:
#         base64_str = image_to_base64(img_path)
#         if base64_str:
#             data_url = f"data:image/jpeg;base64,{base64_str}"
#             history.append((f"{txt} ![]({data_url})", None))
#         else:
#             history.append((txt, "Error processing image"))
#     elif pdf_path:
#         history.append((f"{txt} (PDF uploaded: {pdf_path})", None))
#     elif url_path:
#         history.append((f"{txt} (URL uploaded: {url_path})", None))
#     return history
def add_query_to_history(history, txt, img_path, pdf_path,url_path):
    entry = {"User": txt, "img_path": img_path, "pdf_path": pdf_path, "url_path": url_path}
    if not img_path and not pdf_path and not url_path:
        history.append((txt, None))
    elif img_path:
        base64_str = image_to_base64(img_path)
        if base64_str:
            data_url = f"data:image/jpeg;base64,{base64_str}"
            history.append((f"{txt} ![]({data_url})", None))
            entry["data_url"] = data_url
        else:
            history.append((txt, "Error processing image"))
    elif pdf_path:
        history.append((f"{txt} (PDF uploaded: {pdf_path})", None))
        entry["pdf_path"] = pdf_path

    elif url_path:
        history.append((f"{txt} (URL uploaded: {url_path})", None))
        entry["url_path"] = url_path
    store_history_in_mongo(entry)
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

# def generate_llm_response(history, text, img_path, pdf_path,target_language, app_functionality,url_path):
#     try:
#         if app_functionality == "Chatbot":
#             return handle_chatbot(history, text, target_language)
#         elif app_functionality == "RAG: RAG with upload documents":
#             return handle_rag_gpt(history, text,pdf_path, target_language)
#         elif app_functionality == "Vision-Chatbot":
#             return handle_vision_chatbot(history, text, img_path, target_language)
#         elif app_functionality == "RAG: RAG with processed documents":
#             return handle_preloaded_document_generation(history, text,target_language)
#         elif app_functionality == "Web-Rag-GPT: RAG with the requested website ":
#             return handle_web_rag_gpt(history, text,url_path, target_language)
#         elif app_functionality == "WebRAGQuery: GPT + Duckduckgo search engine + Web RAG pipeline prep + Web Summarizer":
#             return handle_duckduckgo_chatbot(history, text, target_language)
#         elif app_functionality == "Vision-Chatbot: LLAVA":
#             return handle_llava_vision_chatbot(history, text,img_path, target_language)
#         elif app_functionality == "Generate image (stable-diffusion)":
#             return handle_image_generation(history, text)
#     except Exception as e:
#         history.append((None, f"Error generating response: {e}"))
#         return history

def generate_llm_response(history, text, img_path, pdf_path, target_language, app_functionality, url_path):
    response_history = None
    try:
        if app_functionality == "Chatbot":
            response_history = handle_chatbot(history, text, target_language)
        elif app_functionality == "RAG: RAG with upload documents":
            response_history = handle_rag_gpt(history, text, pdf_path, target_language)
        elif app_functionality == "Vision-Chatbot":
            response_history = handle_vision_chatbot(history, text, img_path, target_language)
        elif app_functionality == "RAG: RAG with processed documents":
            response_history = handle_preloaded_document_generation(history, text, target_language)
        elif app_functionality == "Web-Rag: RAG with the requested website":
            response_history = handle_web_rag_gpt(history, text, url_path, target_language)
        elif app_functionality == "Generate image (stable-diffusion)":
            response_history = handle_image_generation(history, text)
        elif app_functionality == "Vision-Chatbot: LLAVA":
            response_history = handle_llava_vision_chatbot(history, text, img_path, target_language)
        elif app_functionality == "WebRAGQuery: GPT + Duckduckgo search engine + Web RAG pipeline prep + Web Summarizer":
            response_history = handle_duckduckgo_chatbot(history, text, target_language)

        if response_history:
            # Store the response history in MongoDB
            # for entry in response_history:
            #     store_history_in_mongo({"Bot": entry[1]})
            last_entry = response_history[-1]
            # print(last_entry)
            store_history_in_mongo({"Bot": last_entry[1]})
    except Exception as e:
        history.append((None, f"Error generating response: {e}"))
        return history

    return history

def handle_chatbot(history, text, target_language):
    response = txt_model.generate_content(text)
    translated_text = translate_text(response.text, target_language)
    history.append((None, translated_text))
    return history

def handle_rag_gpt(history, text,pdf_path, target_language):
    try:
        if not pdf_path:
            history.append((None, translate_text("Please upload a PDF or DOC for RAG-Chatbot functionality.",target_language)))
        else:
            persist_directory = mkdtemp()
            upload_loader_func(pdf_path,persist_directory)
           
            articulation_messages, context_sources = articulation_messages_func(text,persist_directory)
            response = txt_model.generate_content(f"{articulation_messages}")
            bert = evaluate.load("bertscore")
            bert_score = bert.compute(predictions=[response.text], references=[text],model_type="distilbert-base-uncased")
            response = response.text + "\n\n bert Score: " + str(bert_score) + "\n Sources: \n" + context_sources
            # response = txt_model.generate_content(f"{text} (RAG-GPT processing not implemented)")
            translated_text = translate_text(response, target_language)
            history.append((None, translated_text))

    except Exception as e:
        history.append((None, f"Error in RAG-GPT processing: {e}"))
    return history

def handle_web_rag_gpt(history, text, url_path, target_language):
    try:
        if not url_path:
            history.append((None, translate_text("Please upload a URL for Web-RAG-Chatbot functionality.",target_language)))
        else:
            persist_directory = mkdtemp()
            web_upload_loader_func(url_path,persist_directory)

            articulation_messages, context_sources = articulation_messages_func(text,persist_directory)
            print(context_sources)
            print(articulation_messages)
            response = txt_model.generate_content(f"{articulation_messages}")
            print(response.text)
            bert = evaluate.load("bertscore")
            bert_score = bert.compute(predictions=[response.text], references=[text],model_type="distilbert-base-uncased")
            response = response.text + "\n\n bert Score: " + str(bert_score) + "\n Sources: \n" + context_sources
            # response = txt_model.generate_content(f"{text} (RAG-GPT processing not implemented)")
            translated_text = translate_text(response, target_language)
            history.append((None, translated_text))

    except Exception as e:
        history.append((None, f"Error in Web-RAG-GPT processing: {e}"))
    return history

def handle_vision_chatbot(history, text, img_path, target_language):
    if img_path:
        img = PIL.Image.open(img_path)
        response = vis_model.generate_content([text, img])
        translated_text = translate_text(response.text, target_language)
        history.append((None, translated_text))
    else:
        history.append((None, translate_text("Please upload an image for Vision-Chatbot functionality.",target_language)))
    return history

def handle_preloaded_document_generation(history,text,target_language):
    articulation_messages, context_sources = articulation_messages_func(text,"vectordb1")
    response = txt_model.generate_content(f"{articulation_messages}")
    bert = evaluate.load("bertscore")
    bert_score = bert.compute(predictions=[response.text], references=[text],model_type="distilbert-base-uncased")
    response = response.text + "\n\n bert Score: " + str(bert_score) + "\n Sources: \n" + context_sources
    translated_text = translate_text(response, target_language)
    history.append((None, translated_text))
    return history

def handle_duckduckgo_chatbot(history, prompt, target_language):
    articulation_messages, context_sources = articulation_messages_duck_func(prompt)
    print(context_sources)
    print(articulation_messages)
    response = txt_model.generate_content(f"{articulation_messages}")
    print(response.text)
    bert = evaluate.load("bertscore")
    bert_score = bert.compute(predictions=[response.text], references=[prompt],model_type="distilbert-base-uncased")
    response = response.text + "\n\n bert Score: " + str(bert_score) + "\n Sources: \n" + context_sources
    # response = txt_model.generate_content(f"{text} (RAG-GPT processing not implemented)")
    translated_text = translate_text(response, target_language)
    history.append((None, translated_text))
    return history

def handle_image_generation(history, prompt):
    try:
        image = pipe(prompt).images[0]
        image.save("image.png")
        base64_str = image_to_base64_1(image)
        if base64_str:
            data_url = f"data:image/jpeg;base64,{base64_str}"
            history.append((None,f"{prompt} ![]({data_url})"))
    except Exception as e:
        history.append((None, f"Error in image generation: {e}"))
    
    return history

def handle_llava_vision_chatbot(history, image_prompt, img_path, target_language):
    if img_path:
        image = PIL.Image.open(img_path)
        prompt = f"USER: <image>\n{image_prompt} ASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        generate_ids = base_model.generate(**inputs, max_new_tokens=1024)
        decoded_response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        assistant_index = decoded_response.find("ASSISTANT:")
        response = decoded_response[assistant_index + len("ASSISTANT:"):].strip()     
        translated_text = translate_text(response, target_language)
        history.append((None, translated_text))
    else:
        history.append((None, translate_text("Please upload an image for Vision-Chatbot functionality.",target_language)))
    return history

def transcribe_audio_to_text(audio):
        # Load audio file
    waveform, sample_rate = librosa.load(audio, sr=16000)
    # Process the waveform with the processor
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    # Forward pass through the model
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    # Decode the logits into text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    # print(transcription)
    return transcription[0]

def update_message(request: gr.Request):
    return f"Welcome, {request.username}"

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Multimodal Chatbot"):
            with gr.Row() as app_row:
                with gr.Column(scale=1) as left_column:
                    app_functionality = gr.Dropdown(
                        label="Chatbot functionality",
                        choices=[
                            "Chatbot",
                            "Vision-Chatbot",
                            "Vision-Chatbot: LLAVA",
                            "RAG: RAG with upload documents",
                            "RAG: RAG with processed documents",
                            "Web-Rag: RAG with the requested website",
                            "WebRAGQuery: GPT + Duckduckgo search engine + Web RAG pipeline prep + Web Summarizer",
                            "Generate image (stable-diffusion)"
                        ],
                        value="Chatbot",
                        interactive=True,
                    )
                    language_dropdown = gr.Dropdown(
                        label="Select Language",
                        choices=["en", "es", "fr", "de","bn","hi"],
                        value="en",
                        interactive=True,
                    )
                    url_box = gr.Textbox(
                        interactive=True, placeholder="Enter URL", show_label=False
                    )

                    input_audio_block = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Submit your query using voice",
                        waveform_options=gr.WaveformOptions(
                            waveform_color="#01C6FF",
                            waveform_progress_color="#0066B4",
                            skip_length=2,
                            show_controls=True),
                    )
                    audio_submit_btn = gr.Button(value="Submit audio")

                with gr.Column(scale=8) as right_column:
                    with gr.Row() as row_one:
                        with gr.Column(visible=False) as reference_bar:
                            ref_output = gr.Markdown(label="Reference")
                        with gr.Column() as chatbot_output:
                            chatbot = gr.Chatbot(
                                [],
                                elem_id="chatbot",
                                bubble_full_width=False,
                                height=500,
                                avatar_images=(
                                    ("images/user.jpg"), "images/bot.jpeg")
                            )

                    with gr.Row():
                        text_box = gr.Textbox(
                            interactive=True, placeholder="Enter message or upload file....", show_label=False
                        )

                    with gr.Row() as row_two:
                        image_box = gr.Image(type="filepath", label="Upload Image")
                        pdf_box = gr.File(type="filepath", label="Upload PDF")

            txt_msg = text_box.submit(
                fn=add_query_to_history,
                inputs=[chatbot, text_box, image_box, pdf_box, url_box],
                outputs=chatbot,
            ).then(
                fn=generate_llm_response,
                inputs=[chatbot, text_box, image_box, pdf_box, language_dropdown, app_functionality, url_box],
                outputs=chatbot,
            )

            audio_submit_btn.click(
                fn=transcribe_audio_to_text,
                inputs=input_audio_block,
                outputs=text_box
            ).then(
                fn=add_query_to_history,
                inputs=[chatbot, text_box, image_box, pdf_box, url_box],
                outputs=chatbot,
            ).then(
                fn=generate_llm_response,
                inputs=[chatbot, text_box, image_box, pdf_box, language_dropdown, app_functionality, url_box],
                outputs=chatbot,
            )
        with gr.TabItem("Welcome"):
            m = gr.Markdown()
            logout_button = gr.Button("Logout", link="/logout")
            demo.load(update_message, None, m)


    demo.queue()
    demo.launch(auth=[("Soham", "password"), ("Praveen", "password")], debug=True)

