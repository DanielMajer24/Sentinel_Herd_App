import streamlit as st
from PIL import Image
# from transformers import LayoutLMForTokenClassification
# from transformers import LayoutLMTokenizer
# import torch
# import torchvision.transforms as T
# import speech_recognition as sr

import easyocr as ocr
import numpy as np

# from audiorecorder import audiorecorder

import os
import io
# from pydub import AudioSegment
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

@st.cache_resource
def init_models():
    reader = ocr.Reader(['en'], gpu=False, model_storage_directory='.')
    # recognizer = sr.Recognizer()
    # tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    # model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")
    return reader#, recognizer #tokenizer, model

def main():
    
    st.session_state['inspector'] = ''

    st.title('Number OCR App')

    # Load the models
    reader, recognizer = init_models()
    if st.session_state['inspector'] == '':
        inspector = st.text_input("Enter the inspector name", "Enter the inspector name here...")
    else:
        inspector = st.session_state['inspector']

    date = st.date_input("Enter the date", value=None, min_value=None, max_value=None, key=None)
    

    uploaded_file = st.camera_input("Take a picture")
    # uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # transform = T.Compose([
        #     T.Resize((224, 224)),
        #     T.ToTensor(),
        #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])

        # # Apply the transformations to the image
        # image_tensor = transform(image).unsqueeze(0)

        # # Perform inference
        # with torch.no_grad():
        #     outputs = model(image_tensor)

        number_prediction = reader.readtext(image)

        # st.write(result)


        # # Extract the result
        # prediction = outputs.logits.argmax(dim=2).squeeze().tolist()

        # # Convert the prediction to the corresponding character
        # decoded_text = tokenizer.convert_ids_to_tokens(prediction, skip_special_tokens=True)

        # Concatenate characters to get the number
        # number = ''.join(decoded_text).strip()
        number = number_prediction[0][1]
        st.write(f"Detected Number: {number}")

        # Convert the number to an integer if possible

        stt_button = Button(label="Speak", width=100)

        stt_button.js_on_event("button_click", CustomJS(code="""
            var recognition = new webkitSpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
        
            recognition.onresult = function (e) {
                var value = "";
                for (var i = e.resultIndex; i < e.results.length; ++i) {
                    if (e.results[i].isFinal) {
                        value += e.results[i][0].transcript;
                    }
                }
                if ( value != "") {
                    document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
                }
            }
            recognition.start();
            """))

        result = streamlit_bokeh_events(
            stt_button,
            events="GET_TEXT",
            key="listen",
            refresh_on_update=False,
            override_height=75,
            debounce_time=0)

        if result:
            if "GET_TEXT" in result:

                try:
                    text = result.get("GET_TEXT")
                    comments = st.text_area('Animal Comments', text)
                except Exception as e:
                    comments = st.text_area('Animal Comments', "Could not understand the audio!")
                    print("Could not understand the audio!")
            # comments = ""

        with st.form(key='form_1'):
            try:
                integer_number = int(number)
                Sentinel_Animal_Ear_tag = st.number_input("Enter the number", value=integer_number, step=1)

            except ValueError:
                st.write("Unable to convert the detected number to an integer.")

                Sentinel_Animal_Ear_tag = st.number_input("Enter the number", step=1)

            col1, col2, col3 = st.columns(3)
            with col1:
                missing_tag = st.selectbox("Missing Tag", ["Yes", "No"])
            
            with col2:
                sex = st.selectbox("Sex", options=['','M', 'F'], index=0)
            
            with col3:
                BCS_score = st.number_input("BCS Score", value=0, step=1, min_value=0, max_value=5)

            col1, col2, col3 = st.columns(3)

            with col1:
                wound_status = st.selectbox("Wounds", ["", "Yes", "No"], index = 0)
            
            with col2:
                maggots_in_wounds = st.selectbox("Maggots in wounds", options=['','Yes', 'No'], index=0)

            with col3:
                number_of_teeth = st.number_input("Number of teeth", value=0, step=1, min_value=0)

            sample_taken = st.selectbox("Sample taken", ["", "Yes", "No"], index = 0)

            

            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            dict_data = {'inspector':inspector,
                         'date':date.strftime("%Y-%m-%d"),
                         'Sentinel_Animal_Ear_tag':Sentinel_Animal_Ear_tag,
                         'Missing_Tag':missing_tag,
                         'sex':sex,
                         'BCS_score':BCS_score, 
                         'wound_status':wound_status,
                         'maggots_in_wounds':maggots_in_wounds,
                         'number_of_teeth':number_of_teeth,
                         'sample_taken':sample_taken,
                         'comments':comments}
            
            st.write(dict_data)
        


        



if __name__ == "__main__":
    main()  