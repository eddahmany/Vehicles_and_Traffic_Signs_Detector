
import gradio as gr
from utils import predict_gradio_image


imagein = gr.Image(label= "Input image")
imageout = gr.Image(label= "Output image with boxes")
coordinates_out = gr.Text(label= "Founded boxes")

sample_images = [
                 ["images/img1.jpg"],
                 ["images/img2.jpg"],
                 ["images/img3.jpg"],
                 ["images/img4.jpg"],
                 ["images/img5.jpg"],
                 ["images/img6.jpg"]

]
print("hello world")
gr.Interface(
    predict_gradio_image,
    imagein,
    [imageout, coordinates_out],  
    title="Vehicles and Traffic Signs Detection",
    examples=sample_images
).launch()
