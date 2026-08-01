import torch
import streamlit as st

from PIL import Image
from torchvision import transforms

from model import FruitCNN

# Image preprocessing

transform = transforms.Compose([
    transforms.Lambda(
        lambda img: img.convert("RGB")
    ),
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

# Page

st.set_page_config(
    page_title="Produce Inspector",
    page_icon="🔍"
)

st.title("🔍 Produce Inspector")
st.write("Fruit / Vegetable Quality Detector")


# Device

device = torch.device("cpu")


# Load model


@st.cache_resource
def load_model():

    checkpoint = torch.load(
        "best_model.pth",
        map_location=device
    )

    class_names = checkpoint["class_names"]

    model = FruitCNN(
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model, class_names


model, class_names = load_model()


# Nutrition data

PRODUCE_INFO = {

    "apple": {
        "calories": "52 kcal / 100g",
        "sugar": "10.4 g",
        "fun_fact": "Apples are rich in antioxidants."
    },

    "banana": {
        "calories": "89 kcal / 100g",
        "sugar": "12.2 g",
        "fun_fact": "Bananas are berries botanically speaking."
    },

    "bellpepper": {
        "calories": "31 kcal / 100g",
        "sugar": "4.2 g",
        "fun_fact": "Bell peppers contain high vitamin C."
    },

    "carrot": {
        "calories": "41 kcal / 100g",
        "sugar": "4.7 g",
        "fun_fact": "Carrots were originally purple."
    },

    "cucumber": {
        "calories": "15 kcal / 100g",
        "sugar": "1.7 g",
        "fun_fact": "Cucumbers are about 95% water."
    },

    "grape": {
        "calories": "69 kcal / 100g",
        "sugar": "16 g",
        "fun_fact": "Grapes are used to make raisins."
    },

    "guava": {
        "calories": "68 kcal / 100g",
        "sugar": "8.9 g",
        "fun_fact": "Guava contains high vitamin C."
    },

    "mango": {
        "calories": "60 kcal / 100g",
        "sugar": "13.7 g",
        "fun_fact": "Mango is one of the world's most popular fruits."
    },

    "orange": {
        "calories": "47 kcal / 100g",
        "sugar": "9.4 g",
        "fun_fact": "Oranges are technically berries."
    },

    "potato": {
        "calories": "77 kcal / 100g",
        "sugar": "0.8 g",
        "fun_fact": "Potatoes were grown in space experiments."
    },

    "tomato": {
        "calories": "18 kcal / 100g",
        "sugar": "2.6 g",
        "fun_fact": "Tomatoes are botanically fruits."
    },

    "jujube": {
        "calories": "79 kcal / 100g",
        "sugar": "20.2 g",
        "fun_fact": "Jujube tastes like a cross between an apple and a date."
    },

    "pomegranate": {
        "calories": "83 kcal / 100g",
        "sugar": "13.7 g",
        "fun_fact": "A single pomegranate can hold up to 600 seeds."
    },

    "strawberry": {
        "calories": "32 kcal / 100g",
        "sugar": "4.9 g",
        "fun_fact": "Strawberries are the only fruit with seeds on the outside."
    },


}

# Upload image


uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Image"
    )

    image_tensor = transform(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    # Prediction

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    confidence = confidence.item() * 100

    prediction = prediction.item()

    predicted_class = class_names[prediction]

    fruit, freshness = predicted_class.rsplit("_", 1)

    st.success(
        f"Prediction: {freshness.title()} {fruit.title()}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    if confidence < 60:

        st.warning(
            "Low confidence. Try a clearer image."
        )

    info = PRODUCE_INFO.get(
        fruit
    )

    if info:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Calories",
                info["calories"]
            )

        with col2:

            st.metric(
                "Sugar",
                info["sugar"]
            )

        st.subheader(
            "Fun Fact"
        )

        st.write(
            info["fun_fact"]
        )
