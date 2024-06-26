# FaciesSAM: Facies Segment Anything Model

```shell
git clone https://github.com/joshua-atolagbe/FaciesSAM-demo.git
```
# Download Model Checkpoint
```
cd FaciesSAM-demo

```
Download a model checkpoint [here](https://drive.google.com/drive/folders/1uwUPxaNpUfTBIfLldxgFNcdTaoJ8zR7D?usp=sharing).
Create a directory named `model` and move the model here.


### Install the packages:

```shell
pip install -r requirements.txt
```

Install CLIP:

```shell
pip install git+https://github.com/openai/CLIP.git
```


## Web demo

### Gradio demo

- We also provide a UI for testing our method that is built with gradio. You can upload a custom image, select the mode and set the parameters, click the segment button, and get a satisfactory segmentation result. Currently, the UI supports interaction with the 'Everything mode', 'text mode' and 'points mode'. We plan to add support for additional modes in the future. Running the following command in a terminal will launch the demo:

```
python app_gradio.py
```

