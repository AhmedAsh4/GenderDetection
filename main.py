import numpy as np
import tensorflow as tf
import cv2
import face_recognition
import streamlit as st


def loadModel():
    model = tf.keras.models.load_model('MobileNetGenderFineTuneV3.h5')
    return model


def PredImg(img, model):
    image = cv2.imread(img)
    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:
        # Extract coordinates of the face bounding box
        top, right, bottom, left = face_location

        # Extract the face Region of Interest
        face_roi = image[top:bottom, left:right]

        processed_face = cv2.resize(face_roi, (224, 224))  # Resize to match your model's input size

        # Perform gender prediction using your model
        gender_prediction = model.predict(np.expand_dims(processed_face, axis=0))
        gender_prediction = tf.nn.sigmoid(gender_prediction)
        gender_prediction = tf.where(gender_prediction < 0.5, 0, 1)

        if gender_prediction > 0:
            prediction_text = "Female"
        else:
            prediction_text = "Male"

        # Draw a border around the detected face and display the gender prediction
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, prediction_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def imageUpload():
    st.title("Upload Image")
    images = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if images is not None:
        for image in images:
            # Save the uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(image.read())

                # Load the saved image and process it with Pred function
                predImage = PredImg("temp_image.jpg", loadModel())

                # Display the processed image with gender predictions
                st.image(predImage, caption="Processed Image", use_column_width=True)


def main():
    st.set_page_config(layout='wide', page_title="Gender Detection", page_icon='⚥', )
    st.title("Gender Detection Using Deep Learning")
    st.subheader("An Exercise in Transfer Learning and CNNs by Ahmed Ashraf")
    st.markdown("""---""")

    st.subheader("The Website Doesn't Save Any pictures you Upload :)")

    imageUpload()


main()
