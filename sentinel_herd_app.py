import streamlit as st
from PIL import Image
from transformers import LayoutLMForTokenClassification
from transformers import LayoutLMTokenizer
import torch
import torchvision.transforms as T

@st.cache_resource
def init_models():
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")
    return tokenizer, model

def main():
    
    st.title('Number OCR App')

    # Load the models
    tokenizer, model = init_models()

    uploaded_file = st.camera_input("Take a picture")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Apply the transformations to the image
        image_tensor = transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)

        # Extract the result
        prediction = outputs.logits.argmax(dim=2).squeeze().tolist()

        # Convert the prediction to the corresponding character
        decoded_text = tokenizer.convert_ids_to_tokens(prediction, skip_special_tokens=True)

        # Concatenate characters to get the number
        number = ''.join(decoded_text).strip()
        st.write(f"Detected Number: {number}")

        # Convert the number to an integer if possible
        try:
            integer_number = int(number)
            st.write(f"Converted to Integer: {integer_number}")
        except ValueError:
            st.write("Unable to convert the detected number to an integer.")

if __name__ == "__main__":
    main()  