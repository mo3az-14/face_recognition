# Face recognition by moaaz

I tried replicating siamese neural network paper on the LFW dataset. I tried making and training my own model from scratch and fine tuning resnext-50 32x4d model.

**You can try the model and see the blog post [here](https://moaazsfacerecognition.streamlit.app/)**

## project findings

#### My model from scratch

| things I tried         | results                                                                                                                     | conclusion                                        |
| :--------------------- | :-------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------ |
| More trainig           | increasing from 50 -> 100 -> 200 -> 500 all yielded the same result of 80% accuracy on test but and maximum of 90% on train | model is the problem its clearly under fitting    |
| Increase weight decay  | model wouldn't converge for any weight decay values >0.0001                                                                 | our maximum regularization is 0.0001              |
| Add Drop Out layers    | It would only slow down the trainig but wouldn't solve the overfitting issue                                                | Drop out is not really effective here             |
| Increase learning rate | > 0.001 was too much and < 0.00001 was too slow                                                                             | our sweet spot is 0.0001                          |
| Smaller model          | it solves the overfitting but we get only 70% accuracy                                                                      | smaller model is not the solution for overfitting |
| Bigger model           | better training accuracy but still 80% test accuracy                                                                        | more parameters is not the solution               |

Conclusions: the architecture itself is the problem since we can't get over the overfitting problem or else we will underfit

I decided to change my approach in several ways:

- I cropped the faces from the photos and increased the dimensions of the inputs (I used to train on the raw photos of the people )
- I decided to try a model that uses residual connections. I thought it be able to use the data better.
- More aggressive data augmentation
- setup different checkpoints for different metrics like accuracy, precision and recall.

#### Fine tuning resnext 32x4d

I managed to reach 99.95% accuracy on training and 86% on test.

**YES** there is an overfitting problem how did I tackle it ?

- Classifier layer size: I tried using smaller classifiers (one layer) at the end since we have an overfitting problem but it didn't really have an effect.

- Different activation functions: Although I always default to ReLU **but** for some reason sigmoid gave me better results. I was hesitant to use sigmoid due to it's several problems like vanishing gradients, saturation and killing Gradients and slow convergence but I believe I didn't face these problems since I didn't use sigmoid a lot.

- Fewer layers: The resnext-50 32x4d consists of 4 layers so I tried removing the layers one by one to see how it would affect the results. Removing layer 4 decreased training accuracy but didn't have that much effect on the test. However removing any more layers would significantly impact the performance it wouldn't even reach 80% on any metric.

- Increase Drop Out probabilities: it increased the test accuracy 82% -> 86% at 80% drop out. Anything larger than that the model wouldn't converge.

## How to setup the project

1- `pip install -r requirements.txt`

2- run `download_data.py`

3- run `crop_dataset_to_faces.py`

4- you can run `trainer.py` for training on uncropped data or `trainer2.py` for cropped data

folders:

dataset: the directory of original data

cropped_dataset: the directory of data after cropping

logs: this directory contains the model checkpoints

## The end

At the end we are still over fitting with 99% train accuracy and 86% test-accuracy. I believe a huge part of the reason of the overfitting problem is my sampling method used to create the dataset. Also I am using L1 distance to measure the distance between the 2 embeddings which is not really usefull in higher dimensions. Another thing is the use of binary cross entropy loss which is not as effective for modeling the difference between faces as something like triplet loss or ArcFace, I decided to stick with BCE since each one of those would require a lot of change in the code and BCE is what was used in the siamese paper.

I had many different ideas that I can try but this project's purpose was to learn and be comfortable with Pytorch and I spent waaayyyy too much time on this project.

You will find the training recipe in RECIPE.md

## 🔗 Links

gmail : [moaaz2tarik1@gmail.com](moaaz2tarik1@gmail.com)

LinkedIn : [moaaz tarek](https://www.linkedin.com/in/moaaz-tarek/)

GitHub : [moaaz tarek](https://github.com/mo3az-14)

Demo : [https://moaazsfacerecognition.streamlit.app/](https://moaazsfacerecognition.streamlit.app/)

Please reach out to me if you have any tips, courses or anything that can help me learn. thank you!
