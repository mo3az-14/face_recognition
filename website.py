import streamlit as st
from PIL import Image
import config
from model import Model
from torch import nn
from torch import tensor
import torchvision.transforms.v2 as transforms
import torch


def inference(img1: tensor, img2: tensor, threshold: float) -> int:
    return int(torch.sigmoid(model(img1, img2)) > threshold)


img1 = None
img2 = None

# load model
check = torch.load(r"logs\24_07_08_13_10_43___2516b0f0-f000-4e95-a86f-f0e308682eb4.pt")
model = Model()
model.load_state_dict(check["model"])
model.eval()
data_transform = nn.Sequential(
    transforms.ToImage(),
    transforms.ToDtype(
        torch.float32,
        scale=True,
    ),
    transforms.Resize(size=config.IMAGE_SIZE),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
)

st.title("face recognition by moaaz")

text1 = """### Where am *I*?
If you are here that means you want to see my face recognition project, 
yay! If not, how did you get here? This is my machine learning project where
I try tackling face recognition task from scratch. 

:blue[**First**], why not try the project for yourself? 
And don’t worry the data doesn’t get saved, even if I want to, I don’t know how…
"""
st.markdown(text1)

text2 = """### Try it!

Please upload an image of you. This will be used as the source image of you. 
Please choose a good image and don't try to break it because it will break :sweat_smile:
"""
st.markdown(text2)

uploaded_file1 = st.file_uploader(
    label="upload an image of your face to be used as the ground truth",
    type=["png", "jpeg", "jpg"],
    key="img2",
)

if uploaded_file1 is not None:
    ground_truth_image1 = Image.open(uploaded_file1)
    img1 = data_transform(ground_truth_image1)
    img1 = img1.unsqueeze(0)
    st.image(
        ground_truth_image1.resize(config.IMAGE_SIZE, Image.LANCZOS),
        caption="image resized to the actual input dimensions",
    )

text2 = """now you can test with any other image of you or your friend."""
st.markdown(text2)

uploaded_file2 = st.file_uploader(
    label="upload an image to test on", type=["png", "jpeg", "jpg"], key="img1"
)

if uploaded_file2 is not None:
    ground_truth_image2 = Image.open(uploaded_file2)
    img2 = data_transform(ground_truth_image2)
    img2 = img2.unsqueeze(0)
    st.image(
        ground_truth_image2.resize(config.IMAGE_SIZE, Image.LANCZOS),
        caption="image resized to the actual input dimensions",
    )

if img1 is not None and img2 is not None:
    result = inference(img1, img2, 0.5)
    if result == 1:
        st.markdown(
            """this image **is** the same as the ground truth :white_check_mark:"""
        )
    else:
        st.markdown("""this image **is not** the same as the ground truth :x:""")

text3 = """### The project

This project is an attempt to replicate 
[Siamese Neural Networks for One-shot Image Recognition]\
(https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf "Siamese Neural Networks for One-shot Image Recognition") 
paper with the twist of using the [lfw](https://www.kaggle.com/datasets/atulanandjha/lfwpeople "LFW from kaggle") dataset instead of the
[omniglot](https://www.kaggle.com/datasets/qweenink/omniglot "omniglot from kaggle") dataset. The goal of the project is to match the faces of people (a very naive face recognition/lock for the phones). 
The main purpose of this project to me was :red[**learning**]. I learned a lot about :blue[pytorch](90%), :blue[numpy](10%) and
:blue[computer vision] in general in this project. My goal was to try as many different stuff that I could in this project like : 
- Mixed precision training and gradient scailing 
- Creating a custom dataloader
- Expirement tracking 
- Early stopping 
- Learning rate schedulers 
- Weight intialization (although I didn't use it at the end because it produced worse results?)
- Building neural networks from scratch 
- Hyperparameter search and optimization
- Performance analysis 
- Pytorch's transforms api
- A bit of :red[streamlit]
- Profiling my code and model (No need to share results, I just have a slow machine lol) 

and much more.
"""
st.markdown(text3)

text4 = """## Results and numbers

##### Mixed precision training and gradient scailing

According to pytorch's tutorial mixed precision can save memory and offer great speed ups (2~3x). As for the 
speed up, It didn't really speed up the training time on my friend's machine but on my machine (which is 
VERY SLOW in comparison) it cutdown the time from 15 mins/iteration -> 1:30 min./iteration!? yeah I don't really
know why. As for the memory tho it managed to save us 1.8GB of vram which is AMAZING and allowed us to run
more experiments at the same time. Now I know that mixed precision training does have impact on the model's 
performance but as I said before the goal was to try different techniques for fun!

##### Learning rate schedulers  



##### Weight intialization

Well this is a bit of weird one, I tried using xavier normal & uniform intialization but it made the model unable
to learn?. Please send an email me if you have an explaination. 

##### Hyperparameter search and optimization



##### image 



"""
st.markdown(text4)

text5 = """## Who am I?

"""
st.markdown(text5)
