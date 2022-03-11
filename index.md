# Image Deblurring using Deep Convolutional Neural Network
### CSE 455 Final Project by Minh Hoang and Olivia Mirascian
Published on March 10, 2022

## Abstract
In this project, we explore image deblurring techniques and learning rates to unblur differently blurred animated images. To blur the images, we used three techniques: gaussian filter, box filter, and a custom made motion blurring filter. We then used three different convolutional neural network models and two different learning rates to train and compare the resulting images and loss. We discuss more of the takeaways of this project in this video summary here (make link).

## Problem and Motivation
Clear images are important to capture important moments or to record information, although sometimes the images we capture or download result in being blurrier than we imagined. So for our project, we decided to focus on deblurring images that have been blurred in different ways and identify which models and learning rates are most suitable for different types of blurred images. 

## Dataset
We used a kaggle dataset consisting of 92,000 256x256 images of anime character faces without any metadata. We decided to reduce down the size and use 1,000 images from that dataset in our project. We also learned that the models used for deblurring images currently don't work well with colored images so we decided to apply grayscale filter to our images before blurring and training. The dataset is available here (insert link). The dataset doesn't have any specific intended use but we decided to use it since it was consistent in size and content allowing us to extract important information from our deblurring technique. 

## The SRCNN Model
For our model, we used a deep convolutional neural network called SRCNN (super-resolution deep convolutional neural network). 
There are 3 layers to this original model with kernel size 9-1-5. Given a lower resolution image (our blurred image), the first convolutional layer use patch extraction and extracts a set of feature maps, the second layer nonlinearly maps those feature maps to higher resolution patches and the last layer is the reconstruction layer.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/39535587/157801780-67732afb-3c28-4878-8592-63feb44bb51d.png"> 

The paper experiments with different hyper-parameter settings to improve performance. We decided to do similarly and experiment with different depths, kernel size, and learning rates.

## Experiment
For our three models, model 1 was our original model (kernel size 9-1-5), model 2 consisted of a larger kernel and 1 additional layer (9-3-1-5) and model 3 had a smaller kernel with 2 additional layers (9-1-1-1-5)

We first experimented with a learning rate of 0.0001 and an epoch of 20 using all three models:

### Model 1
<img width="300" alt="box_filter_loss_model1" src="https://user-images.githubusercontent.com/39535587/157806246-c5490c52-b009-4798-9410-a184f6b00128.png"> <img width="300" alt="gaussian_loss_model1" src="https://user-images.githubusercontent.com/39535587/157806808-50ccc722-b3ea-4167-8ae3-153ddebe301a.png"> <img width="300" alt="motion_loss_model1" src="https://user-images.githubusercontent.com/39535587/157806878-b1805465-997c-47a6-a2c3-4b4a2c4ac3bb.png"> 

### Model 2
<img width="300" alt="box_filter_loss_model2" src="https://user-images.githubusercontent.com/39535587/157806701-814a8827-c7d0-44ca-80d0-09828c6c236c.png"> <img width="300" alt="gaussian_loss_model2" src="https://user-images.githubusercontent.com/39535587/157806889-9841f232-8879-41b1-9893-007543446661.png"> <img width="300" alt="motion_loss_model2" src="https://user-images.githubusercontent.com/39535587/157806971-114eea4f-084a-48a8-ad76-a4f04dfa7cda.png"> 

### Model 3
<img width="300" alt="box_filter_loss_model3" src="https://user-images.githubusercontent.com/39535587/157806714-1ba85c6e-607a-49ee-95d8-8d428fad3213.png"> <img width="300" alt="gaussian_loss_model3" src="https://user-images.githubusercontent.com/39535587/157806945-4d41604b-6b30-496b-b914-cccae08a3d93.png"> <img width="300" alt="motion_loss_model3" src="https://user-images.githubusercontent.com/39535587/157806907-93393c42-6066-41c1-aa95-633336590e49.png"> 








## References
1. Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. (2014) Learning a Deep Convolutional Network for Image Super-Resolution http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf











Summary: an abstract of your work

Problem Setup: introduction, motivation, project goal

Dataset: what is the data that you used, where did you get it, why did you choose it, did you have to do anything to wrangle it?

Techniques: what techniques (ML/DL Models, algorithms, training parameters, etc)? How did these techniques perform (metrics and evaluation)? Note: the stuff mentioned in this section should be the bulk of your technical work (code and such)

Additional Info: Anything else needed to fully describe your work

References: references to work you consulted









You can use the [editor on GitHub](https://github.com/oliviamirascian/CSE455_FinalWebsite/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/oliviamirascian/CSE455_FinalWebsite/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
