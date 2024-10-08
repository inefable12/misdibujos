import streamlit as st
from inference import text2image, load_pipeline
from io import BytesIO

def app():
    st.header("Text-to-Image Web App")
    st.subheader("Powered by Hugging Face")

    # Entrada de texto del usuario
    user_input = st.text_area(
        "Enter your text prompt below and click the button to submit."
    )

    # Selección del modelo de generación
    option = st.selectbox(
        "Select model (in order of processing time)",
        (
            "nota-ai/bk-sdm-small",
            "CompVis/stable-diffusion-v1-4",
            "runwayml/stable-diffusion-v1-5",
            "prompthero/openjourney",
            "hakurei/waifu-diffusion",
            "stabilityai/stable-diffusion-2-1",
            "dreamlike-art/dreamlike-photoreal-2.0",
        ),
    )

    # Formulario para enviar la solicitud de generación
    with st.form("my_form"):
        submit = st.form_submit_button(label="Submit text prompt")

    # Generar la imagen cuando el usuario hace clic en el botón
    if submit:
        with st.spinner(text="Generating image ... It may take up to 1 hour."):
            im, start, end = text2image(prompt=user_input, repo_id=option)

            # Convertir la imagen en bytes para descargar
            buf = BytesIO()
            im.save(buf, format="PNG")
            byte_im = buf.getvalue()

            # Calcular el tiempo de procesamiento
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)

            st.success(
                "Processing time: {:0>2}:{:0>2}:{:05.2f}.".format(
                    int(hours), int(minutes), seconds
                )
            )

            # Mostrar la imagen generada
            st.image(im)

            # Botón para descargar la imagen generada
            st.download_button(
                label="Click here to download",
                data=byte_im,
                file_name="generated_image.png",
                mime="image/png",
            )

if __name__ == "__main__":
    # Precargar el pipeline para el modelo por defecto para agilizar la primera solicitud
    load_pipeline("stabilityai/stable-diffusion-2-1")
    app()
