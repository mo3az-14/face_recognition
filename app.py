import streamlit as st
from PIL import Image
from model import resnext_50, original_siamese
from torch import nn
import plotly.graph_objects as go
import numpy as np
from torch import tensor
import torchvision.transforms.v2 as transforms
import torch
from facenet_pytorch import MTCNN
import os
from download_models import download_model


@st.cache_resource
def load_model(id, output_path):
    """Load the model, downloading it if necessary."""
    model_path = f"{output_path}"
    if not os.path.exists(model_path):
        download_model(id, output_path)
        st.success(f"{output_path} model downloaded successfully!")
    return model_path


first_model = r"1jTPE_9-AzYMww1BTSTdI7Z64qpSeysoV"
second_model = r"1tSyPKY77sZX5HBkj8-LGdTYbZ_Fi3N11"

st.set_page_config(layout="wide")


global base_check_point, improved_check_point
if not os.path.exists("model.pt"):
    st.write("downloading model")
    base_check_point = load_model(first_model, "model.pt")
else:
    base_check_point = load_model(first_model, "model.pt")

if not os.path.exists("model2.pt"):
    st.write("downloading model")
    improved_check_point = load_model(second_model, "model2.pt")
else:
    improved_check_point = load_model(second_model, "model2.pt")


device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=244, post_process=True, margin=10)


# model inference
def inference(img1: tensor, img2: tensor, threshold: float, model: nn.Module) -> int:
    return int(
        torch.sigmoid(model(img1.to(device).unsqueeze(0), img2.to(device).unsqueeze(0)))
        > threshold
    )


img1 = None
img2 = None
img3 = None
img4 = None

# load weights
checkpoint = torch.load(base_check_point, map_location =device)
checkpoint2 = torch.load(improved_check_point, map_location =device)

# load model from checkpoint
base_model = original_siamese()
base_model.load_state_dict(checkpoint["model"])
base_model.to(device=device)
base_model.eval()

improved_model = resnext_50()
improved_model.load_state_dict(checkpoint2["model"]["best_accuracy_model"])
improved_model.to(device=device)
improved_model.eval()

# load training metrics
accuracy = checkpoint["accuracy"]
precision = checkpoint["precision"]
epochs = checkpoint["epochs"]
recall = checkpoint["recall"]
fscore = checkpoint["fscore"]
train_loss, test_loss = checkpoint["train_loss"], checkpoint["test_loss"]

# improved model accuracy
improved_test_accuracy = checkpoint2["test_accuracy"]
improved_test_precision = checkpoint2["test_precision"]
improved_test_recall = checkpoint2["test_recall"]
improved_test_fscore = checkpoint2["test_fscore"]
improved_train_accuracy = checkpoint2["train_accuracy"]
improved_train_precision = checkpoint2["train_precision"]
improved_train_recall = checkpoint2["train_recall"]
improved_train_fscore = checkpoint2["train_fscore"]
improved_train_loss, improved_test_loss = (
    checkpoint2["train_loss"],
    checkpoint2["test_loss"],
)
improved_epochs = checkpoint2["epochs"]

# transformations
data_transform1 = nn.Sequential(
    transforms.ToImage(),
    transforms.ToDtype(
        torch.float32,
        scale=True,
    ),
    transforms.Resize(size=(128, 128)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
)
data_transform2 = nn.Sequential(
    transforms.ToImage(),
    transforms.ToDtype(
        torch.float32,
        scale=True,
    ),
)

st.sidebar.title("Sections")
st.sidebar.markdown("Part 1")
st.sidebar.markdown("[Where am I?](#WhereamI?)")
st.sidebar.markdown("[Try it!](#Tryit!)")
st.sidebar.markdown("[The project](#Theproject)")
st.sidebar.markdown("[Some of the things that I liked](#SomeofthethingsthatIliked)")
st.sidebar.markdown("[graphs](#graphs)")
st.sidebar.markdown("[Bad results](#Badresults?)")
st.sidebar.markdown("part 2")
st.sidebar.markdown("[part 2](#part2)")
st.sidebar.markdown("[graphs](#graphs2)")
st.sidebar.markdown("[Try it!](#Tryit1!)")
st.sidebar.markdown("[Contact me](#Contactme)")

# styles for the adding color to headings
st.markdown(
    """
    <style>
    .main_title {
        font-size:48px;
        color: #fffff;
        padding: 0 ;
        margin_bottom: 0;
        text-align: center;
    }
    .subtitle {
        font-size: 40px;
        color: #13C4A3;
    }
    .subsub{
        font-size: 25px;
        color: #C7F9F0;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Title of the page
st.markdown(
    """<div class='main_title'>face recognition by moaaz</div>""",
    unsafe_allow_html=True,
)

# Where am I section
text1 = """<div class='subtitle' id="WhereamI?" >Where am I? </div>
If you are here that means you want to see my face recognition project, 
yay! If not, how did you get here? This is my machine learning project where
I try tackling face recognition task from scratch. This project is (or was depending on when are you reading this) a 2 parts project. 
part 1 I replicate the Siamese Neural Networks for One-shot Image Recognition and figure out the architecture, dataloader, training etc...
and in part 2 I try using a bigger more complex model to get better results on the dataset of my choice (LFW).

**First**, why not try the project for yourself? 
And don’t worry the data doesn’t get saved, even if I want to, I don’t know how…
"""
st.markdown(text1, unsafe_allow_html=True)

# Try it section
text2 = """<div class='subtitle' id="Tryit!">Try it!</div>

Please upload an image of you. This will be used as the source image. 
Please choose a good image and don't try to break it because it will break :sweat_smile:
"""
st.markdown(text2, unsafe_allow_html=True)

# upload photos
uploaded_file1 = st.file_uploader(
    label="upload an image of your face to be used as the ground truth",
    type=["png", "jpeg", "jpg"],
    key="img2",
)

# if a file is uploaded apply transformations required and display it
if uploaded_file1 is not None:
    ground_truth_image1 = Image.open(uploaded_file1)
    img1 = data_transform1(ground_truth_image1.convert("RGB"))
    st.image(
        ground_truth_image1.resize((128, 128), Image.LANCZOS),
        caption="image resized to the actual input dimensions",
    )

text3 = """now you can test with any other image of you or your friend."""
st.markdown(text3)

uploaded_file2 = st.file_uploader(
    label="upload an image to test on", type=["png", "jpeg", "jpg"], key="img1"
)

if uploaded_file2 is not None:
    ground_truth_image2 = Image.open(uploaded_file2)
    img2 = data_transform1(ground_truth_image2.convert("RGB"))
    st.image(
        ground_truth_image2.resize((128, 128), Image.LANCZOS),
        caption="image resized to the actual input dimensions",
    )

if img1 is not None and img2 is not None:
    result = inference(img1, img2, 0.5, base_model)
    if result == 1:
        st.markdown(
            """this image **is** the same as the ground truth :white_check_mark:"""
        )
    else:
        st.markdown("""this image **is not** the same as the ground truth :x:""")

# The project section
text4 = """<div class='subtitle' id="Theproject">The project</div>

This project is an attempt to replicate 
[Siamese Neural Networks for One-shot Image Recognition]\
(https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf "Siamese Neural Networks for One-shot Image Recognition") 
paper with some changes like using the [lfw](https://www.kaggle.com/datasets/atulanandjha/lfwpeople "LFW from kaggle") dataset instead of the
[omniglot](https://www.kaggle.com/datasets/qweenink/omniglot "omniglot from kaggle") dataset and using Adam optimizer instead of SGD (I tried it but adam converged faster ).
The goal of the project is to match the faces of people (a very naive face recognition/lock for the phones).The main purpose of this 
project to me was **learning**.I learned a lot about pytorch (90%), numpy (10%) and computer vision in general in this project. 
My goal was to get my hands dirty with pytorch and try replicating a paper.

Here is a list of stuff I used/tried in this project:
- Building a neural network from scratch
- Hyperparameter search and optimization
- Mixed precision training and gradient scailing 
- Creating a custom dataloader
- Expirement tracking 
- Early stopping 
- Learning rate schedulers 
- Weight intialization (although I didn't use it at the end because it produced worse results?)
- Performance analysis 
- Pytorch's transforms api
- A bit of streamlit
- Profiling my code and model using pytorch profiler (No need to share results, I just have a slow machine lol) 

and much more.
"""
st.markdown(text4, unsafe_allow_html=True)

# Some of the things I liked section
text5 = """<div class='subtitle' id="SomeofthethingsthatIliked">Some of the things that I liked</div>

<div class='subsub'> Mixed precision training and gradient scailing</div>

According to pytorch's tutorial mixed precision can save memory and offer great speed ups (2~3x). For the 
speed up, It did really speed up the training time on my friend's machine by 60%. As for the memory, it managed to save us ~30%
or 512mb of vram which allowed us to run more experiments at the same time. I was worried about the whole effect on performance thing
but it turned out to be negligble in my case.

<div class='subsub'>Learning rate schedulers</div>

This was my first time using Learning rate schedulers and they really speed up the convergece of the model. I tried different combinations
of learning rate schedulers ( StepLR & CosineAnnealingLR ) and optimizers ( Adam & SGD ). I found using Adam and with StepLR allowed for 
much faster convergence despite the slow start. SGD worked best with CosineAnnealingLR but required much higher learning rate.
Results of my expirements showed that in my use case despite SGD having faster start than Adam but Adam converged much faster than SGD 
later on. 

<div class='subsub'>Weight intialization </div>

Well this is a bit of weird one, I tried using xavier normal & uniform intialization but it made the model unable
to learn?. Please send an email me if you have an explaination. 

"""
st.markdown(text5, unsafe_allow_html=True)
text8 = """<div class='subtitle' id="graphs">Graphs</div>"""
st.markdown(text8, unsafe_allow_html=True)

# charts
# loss chart
loss_fig = go.Figure()
loss_fig.add_trace(
    go.Scatter(
        name="train_loss",
        x=np.arange(1, epochs + 1),
        y=train_loss,
        mode="lines",
        marker={"color": "rgb(72, 169, 166)"},
    )
)
loss_fig.add_trace(
    go.Scatter(
        name="test_loss",
        x=np.arange(1, epochs + 1),
        y=test_loss,
        mode="lines",
        marker={"color": "rgb(161, 195, 73)"},
    )
)
loss_fig_config = {
    "title": {"text": "training vs test loss"},
    "xaxis": {"showgrid": False},
    "yaxis": {"range": [0, max(train_loss, test_loss)]},
}
loss_fig.update_layout(loss_fig_config)
st.plotly_chart(loss_fig)

# accuracy chart
accuracy_fig = go.Figure()
accuracy_fig.add_trace(
    go.Scatter(
        x=np.arange(1, epochs + 1),
        y=accuracy,
        mode="lines",
        marker={"color": "rgb(72, 169, 166)"},
    )
)
accuracy_fig_config = {
    "title": {"text": "test accuracy "},
    "xaxis_title": {
        "text": "(calculated every 2 epochs)",
        "font": {"color": "rgb(184, 184, 184)"},
    },
    "xaxis": {"showgrid": False},
    "yaxis": {"showgrid": True, "range": [0, 1]},
}
accuracy_fig.update_layout(accuracy_fig_config)
st.plotly_chart(accuracy_fig)

# precision chart
precision_fig = go.Figure()
precision_fig.add_trace(
    go.Scatter(
        x=np.arange(1, epochs + 1),
        y=np.array(precision)[:, 0],
        mode="lines",
        marker={"color": "rgb(72, 169, 166)"},
    )
)
precision_fig_config = {
    "title": {"text": "test precision"},
    "xaxis_title": {
        "text": "(calculated every 2 epochs)",
        "font": {"color": "rgb(184, 184, 184)"},
    },
    "xaxis": {"showgrid": False},
    "yaxis": {"showgrid": True, "range": [0, 1]},
}
precision_fig.update_layout(precision_fig_config)
st.plotly_chart(precision_fig)

# recall chart
recall_fig = go.Figure()
recall_fig.add_trace(
    go.Scatter(
        x=np.arange(1, epochs + 1),
        y=np.array(recall)[:, 0],
        mode="lines",
        marker={"color": "rgb(72, 169, 166)"},
    )
)
recall_fig_config = {
    "title": {"text": "test recall"},
    "xaxis_title": {
        "text": "(calculated every 2 epochs)",
        "font": {"color": "rgb(184, 184, 184)"},
    },
    "xaxis": {"showgrid": False},
    "yaxis": {"showgrid": True, "range": [0, 1]},
}
recall_fig.update_layout(recall_fig_config)
st.plotly_chart(recall_fig)

# fscore chart
fscore_fig = go.Figure()
fscore_fig.add_trace(
    go.Scatter(
        x=np.arange(1, epochs + 1),
        y=np.array(fscore)[:, 0],
        mode="lines",
        marker={"color": "rgb(72, 169, 166)"},
    )
)
fscore_fig_config = {
    "title": {"text": "test fscore"},
    "xaxis_title": {
        "text": "(calculated every 2 epochs)",
        "font": {"color": "rgb(184, 184, 184)"},
    },
    "xaxis": {"showgrid": False},
    "yaxis": {"showgrid": True, "range": [0, 1]},
}
fscore_fig.update_layout(fscore_fig_config)
st.plotly_chart(fscore_fig, **{"config": fscore_fig_config})

text6 = """<div class='subtitle' id="Badresults?">Bad results?</div>

Well the results aren't impressive and frankly we get only 80% accuracy on test data which is nothing compared to the current state of 
the art models which reach 99.90% accuracy but the goal of the project was to try to replicate a paper that I've never read before without 
using or copying an already made implementation (I didn't use chatgpt becuase It will just spit out the answer). But I don't like the results
so in part 2 I will try to increase the performance. I don't want to set any expectations but I hope that by the time you open this project
again it will have been updated with better results at the end. 

If for some reason you reached here **thank you** for reading and come check again in another week or two when the project get's updated
with better results :heart:.

"""

st.markdown(text6, unsafe_allow_html=True)
text7 = """<div class='subtitle' id="part2">Part 2</div>

I tried several things to improve performance.

1- More data augmentation: because there was a clear overfitting problem 

2- more regularization by adding dropout layers 

3- better backbone: I fine-tuned pretrained resnext50_32x4d model (default weights from pytorch) which did indeed improve the performance
and require way less training (100->50 epochs)

4- longer training: I tried training for 50,100,150,200 epochs and anything above 40 epochs wouldn't improve the
performance  

*Still* we have an overfitting problem. Solutions? 

1- More regularization: I increased the weight decay but anything above 0.0001 the model wouldn't converge

2- Smaller model: 

The way I fine-tuned the model is by removing the last fully connected layer and adding a small
Linear layer (embedding) right before the final classifier that will decide if the 2 images are similar or not. 
The plan was to make it so the we can compute the embeddings with the embedding layer and store it somewhere (database), then during
inference we would only use the last small linear classifier to pick the matching picture.  

"""
st.markdown(text7, unsafe_allow_html=True)

text8 = """<div class='subtitle' id="graphs2">Graphs</div>"""
st.markdown(text8, unsafe_allow_html=True)

# charts
# loss chart
loss_fig = go.Figure()
loss_fig.add_trace(
    go.Scatter(
        name="train_loss",
        x=np.arange(1, improved_epochs + 1),
        y=improved_train_loss,
        mode="lines",
        marker={"color": "rgb(72, 169, 166)"},
    )
)
loss_fig.add_trace(
    go.Scatter(
        name="test_loss",
        x=np.arange(1, improved_epochs + 1),
        y=improved_test_loss,
        mode="lines",
        marker={"color": "rgb(161, 195, 73)"},
    )
)
loss_fig_config = {
    "title": {"text": "training loss vs test loss"},
    "xaxis": {"showgrid": False},
    "yaxis": {
        "showgrid": True,
        "range": [0, max(improved_test_loss, improved_train_accuracy)],
    },
}
loss_fig.update_layout(loss_fig_config)
st.plotly_chart(loss_fig)

# accuracy chart
accuracy_fig = go.Figure()
accuracy_fig.add_trace(
    go.Scatter(
        name="train accuracy",
        x=np.arange(1, improved_epochs + 1),
        y=improved_train_accuracy,
        mode="lines",
        marker={"color": "rgb(72, 169, 166)"},
    )
)
accuracy_fig.add_trace(
    go.Scatter(
        name="test accuracy",
        x=np.arange(1, improved_epochs + 1),
        y=improved_test_accuracy,
        mode="lines",
        marker={"color": "rgb(161, 195, 73)"},
    )
)
accuracy_fig_config = {
    "title": {"text": "accuracy "},
    "xaxis_title": {
        "font": {"color": "rgb(184, 184, 184)"},
    },
    "xaxis": {"showgrid": False},
    "yaxis": {"showgrid": True, "range": [0, 1]},
}
accuracy_fig.update_layout(accuracy_fig_config)
st.plotly_chart(accuracy_fig)

# precision chart
precision_fig = go.Figure()
precision_fig.add_trace(
    go.Scatter(
        name="train precision",
        x=np.arange(1, improved_epochs + 1),
        y=np.array(improved_train_precision),
        mode="lines",
        marker={"color": "rgb(72, 169, 166)"},
    )
)
precision_fig.add_trace(
    go.Scatter(
        name="test precision",
        x=np.arange(1, improved_epochs + 1),
        y=np.array(improved_test_precision),
        mode="lines",
        marker={"color": "rgb(161, 195, 73)"},
    )
)

precision_fig_config = {
    "title": {"text": "precision"},
    "xaxis_title": {
        "font": {"color": "rgb(184, 184, 184)"},
    },
    "xaxis": {"showgrid": False},
    "yaxis": {"showgrid": True, "range": [0, 1]},
}
precision_fig.update_layout(precision_fig_config)
st.plotly_chart(precision_fig)

# recall chart
recall_fig = go.Figure()
recall_fig.add_trace(
    go.Scatter(
        name="train recall",
        x=np.arange(1, improved_epochs + 1),
        y=np.array(improved_train_recall),
        mode="lines",
        marker={"color": "rgb(72, 169, 166)"},
    )
)
recall_fig.add_trace(
    go.Scatter(
        name="test recall",
        x=np.arange(1, improved_epochs + 1),
        y=np.array(improved_test_recall),
        mode="lines",
        marker={"color": "rgb(161, 195, 73)"},
    )
)
recall_fig_config = {
    "title": {"text": "recall"},
    "xaxis_title": {
        "font": {"color": "rgb(184, 184, 184)"},
    },
    "xaxis": {"showgrid": False},
    "yaxis": {"showgrid": True, "range": [0, 1]},
}
recall_fig.update_layout(recall_fig_config)
st.plotly_chart(recall_fig)

# fscore chart
fscore_fig = go.Figure()
fscore_fig.add_trace(
    go.Scatter(
        name="train fscore",
        x=np.arange(1, improved_epochs + 1),
        y=np.array(improved_train_fscore),
        mode="lines",
        marker={"color": "rgb(72, 169, 166)"},
    )
)
fscore_fig.add_trace(
    go.Scatter(
        name="test fscore",
        x=np.arange(1, improved_epochs + 1),
        y=np.array(improved_test_fscore),
        mode="lines",
        marker={"color": "rgb(161, 195, 73)"},
    )
)
fscore_fig_config = {
    "title": {"text": "fscore"},
    "xaxis_title": {
        "font": {"color": "rgb(184, 184, 184)"},
    },
    "xaxis": {"showgrid": False},
    "yaxis": {"showgrid": True, "range": [0, 1]},
}
fscore_fig.update_layout(fscore_fig_config)
st.plotly_chart(fscore_fig, **{"config": fscore_fig_config})

text9 = """<div class='subtitle' id="Tryit1!">Try it!</div>

Please upload an image of you. This will be used as the source image. 
"""
st.markdown(text9, unsafe_allow_html=True)

# upload photos
uploaded_file3 = st.file_uploader(
    label="upload an image of your face to be used as the ground truth",
    type=["png", "jpeg", "jpg"],
    key="img3",
)

# if a file is uploaded apply transformations required and display it
if uploaded_file3 is not None:
    ground_truth_image3 = Image.open(uploaded_file3)
    img3 = data_transform2(ground_truth_image3.convert("RGB").resize((244, 244)))
    img3, prob = mtcnn(ground_truth_image3.convert("RGB"), return_prob=True)
    st.image(
        ground_truth_image3.resize((244, 244), Image.LANCZOS),
        caption="image resized to the actual input dimensions",
    )

text10 = """now you can test with any other image of you or your friend."""
st.markdown(text10)

uploaded_file4 = st.file_uploader(
    label="upload an image to test on", type=["png", "jpeg", "jpg"], key="img4"
)

if uploaded_file4 is not None:
    ground_truth_image4 = Image.open(uploaded_file4).convert("RGB")
    img4 = mtcnn(ground_truth_image4)
    st.image(
        ground_truth_image4.resize((244, 244), Image.LANCZOS),
        caption="image resized to the actual input dimensions",
    )
if img3 is not None and img4 is not None:
    result = inference(img3, img4, 0.5, improved_model)
    if result == 1:
        st.markdown(
            """this image **is** the same as the ground truth :white_check_mark:"""
        )
    else:
        st.markdown("""this image **is not** the same as the ground truth :x:""")

# contact me section
text11 = """<div class='subtitle' id="Contactme">contact me</div>

gmail : moaaz2tarik1@gmail.com

Linkedin: [https://www.linkedin.com/in/moaaz-tarek/](https://www.linkedin.com/in/moaaz-tarek/ "moaaz tarek")

github: [https://github.com/mo3az-14](https://github.com/mo3az-14 "moaaz tarek")

"""
st.markdown(text11, unsafe_allow_html=True)
