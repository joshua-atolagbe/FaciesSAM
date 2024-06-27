# FaciesSAM: Facies Segment Anything Model

This directory contain instruction on how to run FaciesSAM+CLIP. This will help you to interact with seismic images via prompt guidance. 

## Run web demo

 We provide a UI for testing our method that is built with gradio. You can upload a custom image, select the mode and set the parameters, click the segment button, and get a satisfactory segmentation result. Currently, the UI supports interaction with the `Everything mode`, `text mode` and `points mode`. We plan to add support for additional modes in the future. Running the following command in a terminal will launch the demo:

```python3
python app_gradio.py
```

