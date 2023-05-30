import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

# Load models and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Define Streamlit app
def main():
    st.title("Image Captioning")

    # Upload image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process image and generate caption
        caption = predict_step([image])
        st.write("Caption:", caption)


def predict_step(images):
    images_list = []

    for image in images:
        i_image = Image.open(image)

        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images_list.append(i_image)

    # Extract features
    pixel_values = feature_extractor(images=images_list, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate caption
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds[0]


if __name__ == "__main__":
    main()
