from ultralytics import YOLO
import gradio as gr
import torch
import numpy as np
from utils.tools_gradio import fast_process
from utils.tools import format_results, box_prompt, point_prompt, text_prompt
from PIL import ImageDraw
from fastsam import FastSAM

from warnings import filterwarnings
filterwarnings("ignore")


torch.manual_seed(2024)

# Load the pre-trained model
model = FastSAM('models/FaciesSAM-x-640.pt') #modify based on model

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Title text (without the logo - will be added separately)
title_text = """
<center>
    <strong><font size="8">🌍 User-Guided Approach for Seismic Facies Segmentation</font></strong>
    <p><strong>Creators: Joshua Atolagbe & Ardiansyah Koeshidayatullah</strong></p>
    <p><em>(Paleo3, REservoir, DIagenesis Characterization and Technology (PREDICT) Research Group, KFUPM)</em></p>
</center>
"""

news = """ # 📖 Potential
        🔥 Reformulating seismic facies segmentation as a Segment All or Segment One (SASO) task     
        
        🔥 Focused and localized seismic facies analysis

        🔥 Human-in-the-loop seismic interpretation   
        """  

description_e = """This is a demo of FaciesSAM
                
                🎯 Upload an Image, segment it with FaciesSAM (Everything model). The other modes will come soon.
                
                ⌛️ It takes about ~0.5s second to generate segment results. The concurrency_count of queue is 1, please wait for a moment when it is crowded.
                
                🚀 To get faster results, you can use a smaller input size and leave high_visual_quality unchecked.
                
                
              """

description_p = """ # 🎯 Instructions for points mode
                This is a demo of FaciesSAM
                
                1. Upload an image or choose an example.
                
                2. Choose the point label ('Add mask' means a positive point. 'Remove' Area means a negative point that is not segmented).
                
                3. Add points one by one on the image.
                
                4. Click the 'Segment with points prompt' button to get the segmentation results.
                
                **5. If you get Error, click the 'Clear points' button and try again may help.**
                
              """

examples = [["app/examples/test_image_inline_200.jpg"], ["app/examples/test_image_inline_216.jpg"], 
	    ["app/examples/test_image_inline_230.jpg"], ["app/examples/test_image_inline_235.jpg"],
	    ["app/examples/test_image_inline_299.jpg"]]

default_example = examples[-1]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def segment_everything(
    input,
    input_size=640, 
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=False,
    use_retina=True,
    text="",
    wider=False,
    mask_random_color=True,
):
    input_size = int(input_size)  # Ensure imgsz is an integer
    # Thanks for the suggestion by hysts in HuggingFace.
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size,)

    if len(text) > 0:
        results = format_results(results[0], 0)
        annotations, _ = text_prompt(results, text, input, device=device, wider=wider)
        annotations = np.array([annotations])
    else:
        annotations = results[0].masks.data
    
    fig = fast_process(annotations=annotations,
                       image=input,
                       device=device,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours,)
    return fig


def segment_with_points(
    input,
    input_size=640, 
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=False,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label
    
    input_size = int(input_size)  # Ensure imgsz is an integer
    # Thanks for the suggestion by hysts in HuggingFace.
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))
    
    scaled_points = [[int(x * scale) for x in point] for point in global_points]

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size,)
    
    results = format_results(results[0], 0)
    annotations, _ = point_prompt(results, scaled_points, global_point_label, new_h, new_w)
    annotations = np.array([annotations])

    fig = fast_process(annotations=annotations,
                       image=input,
                       device=device,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours,)

    global_points = []
    global_point_label = []
    return fig, None


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 8, (255, 255, 0) if label == 'Add Mask' else (255, 0, 255)
    global_points.append([x, y])
    global_point_label.append(1 if label == 'Add Mask' else 0)
    
    print(x, y, label == 'Add Mask')
    
    # Create a drawing object
    draw = ImageDraw.Draw(image)
    draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)
    return image


cond_img_e = gr.Image(label="Input", value=default_example[0], type='pil')
cond_img_p = gr.Image(label="Input with points", value=default_example[0], type='pil')
cond_img_t = gr.Image(label="Input with text", value=default_example[0], type='pil')

segm_img_e = gr.Image(label="Segmented Image", interactive=False, type='pil')
segm_img_p = gr.Image(label="Segmented Image with points", interactive=False, type='pil')
segm_img_t = gr.Image(label="Segmented Image with text", interactive=False, type='pil')

global_points = []
global_point_label = []

input_size_slider = gr.components.Slider(minimum=256,
                                         maximum=1024,
                                         value=640,
                                         step=64,
                                         label='Input_size',
                                         info='Our model was trained on a size of 320 and 640')

with gr.Blocks(css=css, title='Facies Segment Anything') as demo:
    with gr.Row():
        with gr.Column(scale=1):
            logo_html = gr.HTML('<img src="/file=app/logo.svg" alt="FaciesSAM Logo" width="200">')
            
            # Add the rest of the title content
            gr.Markdown(title_text)

        with gr.Column(scale=1):
            # News
            gr.Markdown(news)

    with gr.Tab("Everything mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_e.render()

            with gr.Column(scale=1):
                segm_img_e.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                input_size_slider.render()

                with gr.Row():
                    contour_check = gr.Checkbox(value=True, label='withContours', info='draw the edges of the masks')

                    with gr.Column():
                        segment_btn_e = gr.Button("Segment Everything", variant='primary')
                        clear_btn_e = gr.Button("Clear", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(examples=examples,
                            inputs=[cond_img_e],
                            outputs=segm_img_e,
                            fn=segment_everything,
                            # cache_examples=True,
                            examples_per_page=4)

            with gr.Column():
                with gr.Accordion("Advanced options", open=False):
                    iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label='iou', info='iou threshold for filtering the annotations')
                    conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label='conf', info='object confidence threshold')
                    with gr.Row():
                        mor_check = gr.Checkbox(value=False, label='better_visual_quality', info='better quality using morphologyEx')
                        with gr.Column():
                            retina_check = gr.Checkbox(value=True, label='use_retina', info='draw high-resolution segmentation masks')

                # Description
                gr.Markdown(description_e)

    segment_btn_e.click(segment_everything,
                        inputs=[
                            cond_img_e,
                            input_size_slider,
                            iou_threshold,
                            conf_threshold,
                            mor_check,
                            contour_check,
                            retina_check,
                        ],
                        outputs=segm_img_e)

    with gr.Tab("Points mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_p.render()

            with gr.Column(scale=1):
                segm_img_p.render()
                
        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    add_or_remove = gr.Radio(["Add Mask", "Remove Area"], value="Add Mask", label="Point_label (foreground/background)")

                    with gr.Column():
                        segment_btn_p = gr.Button("Segment with points prompt", variant='primary')
                        clear_btn_p = gr.Button("Clear points", variant='secondary')

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(examples=examples,
                            inputs=[cond_img_p],
                            # outputs=segm_img_p,
                            # fn=segment_with_points,
                            # cache_examples=True,
                            examples_per_page=4)

            with gr.Column():
                # Description
                gr.Markdown(description_p)

    cond_img_p.select(get_points_with_draw, [cond_img_p, add_or_remove], cond_img_p)

    segment_btn_p.click(segment_with_points,
                        inputs=[cond_img_p],
                        outputs=[segm_img_p, cond_img_p])

    with gr.Tab("Text mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_t.render()

            with gr.Column(scale=1):
                segm_img_t.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                input_size_slider_t = gr.components.Slider(minimum=256,
                                                           maximum=1024,
                                                           value=640,
                                                           step=64,
                                                           label='Input_size',
                                                           info='Our model was trained on a size of 320 and 640')
                with gr.Row():
                    with gr.Column():
                        contour_check = gr.Checkbox(value=True, label='withContours', info='draw the edges of the masks')
                        text_box = gr.Textbox(label="text prompt", value="middle ns facies")

                    with gr.Column():
                        segment_btn_t = gr.Button("Segment with text", variant='primary')
                        clear_btn_t = gr.Button("Clear", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(examples=examples,
                            inputs=[cond_img_e],
                            # outputs=segm_img_e,
                            # fn=segment_everything,
                            # cache_examples=True,
                            examples_per_page=4)

            with gr.Column():
                with gr.Accordion("Advanced options", open=False):
                    iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label='iou', info='iou threshold for filtering the annotations')
                    conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label='conf', info='object confidence threshold')
                    with gr.Row():
                        mor_check = gr.Checkbox(value=False, label='better_visual_quality', info='better quality using morphologyEx')
                        retina_check = gr.Checkbox(value=True, label='use_retina', info='draw high-resolution segmentation masks')
                        wider_check = gr.Checkbox(value=False, label='wider', info='wider result')

                # Description
                gr.Markdown(description_e)
    
    gr.HTML("""
    <div class="footer">
        <p>
            <center>PREDICT Research Group, KFUPM © 2025</center>
        </p>
    </div>
    """)

    segment_btn_t.click(segment_everything,
                        inputs=[
                            cond_img_t,
                            input_size_slider_t,
                            iou_threshold,
                            conf_threshold,
                            mor_check,
                            contour_check,
                            retina_check,
                            text_box,
                            wider_check,
                        ],
                        outputs=segm_img_t)

    def clear():
        return None, None
    
    def clear_text():
        return None, None, None

    clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])
    clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])
    clear_btn_t.click(clear_text, outputs=[cond_img_p, segm_img_p, text_box])

demo.queue()
demo.launch(share=True, allowed_paths=["app/logo.svg"])