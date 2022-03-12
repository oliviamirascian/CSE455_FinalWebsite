# <div align="center">Anime Faces Deblurring using Deep Convolutional Neural Network </div>
### <div align="center"> Minh Hoang, Olivia Mirascian </div>
<div align="right"> Published on March 11, 2022 </div>

## Abstract
In this project, we explore image deblurring techniques and learning rates to unblur differently blurred animated images. To blur the images, we used three techniques: gaussian filter, box filter, and a custom made motion blurring filter. We then used three different deep convolutional neural network models with some different learning rates to train and compare the resulting images and loss. We discuss more of the takeaways of this project in this video summary here (make link).

## Introduction
Clear images are important to capture important moments or to record information, although sometimes the images we capture or download result in being blurrier than we imagined. In low-level Computer Vision, we've learned different methods to blur a clear image, as well as different blurring filters. Moving on to higher-level Computer Vision, we were introduced with more complex techniques and models that excel in multiple tasks such as object detection, semantic segmentation, image classification,... We were particularly interested GANs (Generative Adversarial Networks) and one of its applications, which is Image Super-Resolution. 

Therefore, for our project, we decided to re-implement Dong *et al.*'s [paper<sup>[1]</sup>](https://ieeexplore.ieee.org/document/7115171) on Image Super-Resolution. This problem was initially implemented using MATLAB and Keras [here](https://github.com/YapengTian/SRCNN-Keras), so we will re-implement it using PyTorch on a slightly simpler tasks: Deblurring images that have been blurred in different ways and identify which models and learning rates are most suitable for different types of blurred images. Our code, plots and results can be found [here](https://github.com/oliviamirascian/CSE455_finalproject).

## Dataset and Pre-processing
We used a Kaggle dataset consisting of 92,219 256x256 images of anime character faces without any metadata. For training time purpose, we decided to reduce down the size and use 1,000 images from that dataset in our project. We also learned  that the models used in the paper didn't work too well with colored images so we decided to apply grayscale filter to our images before blurring and training. The dataset doesn't have any specific intended use but we decided to use it since it was consistent in size and content allowing us to extract important information from our deblurring technique. The dataset is available [here](https://www.kaggle.com/scribbless/another-anime-face-dataset).

<div align="center">
<figure>

 <img alt="sample1" src="https://user-images.githubusercontent.com/39535587/157968360-4547c5ec-358f-4877-acfc-a84f209d23bb.jpg"> 
 <img alt="sample2" src="https://user-images.githubusercontent.com/39535587/157968463-42947e2b-afd8-4857-b909-6cfadae8ec39.jpg">
 <img alt="sample3" src="https://user-images.githubusercontent.com/39535587/157968979-1d474ade-c196-43cb-ac7c-a039e37334d6.jpg">
 
</figure>
</div>



## Model Selection: SRCNN Model
For our model selection, we used the deep convolutional neural network model proposed in the paper, which is called SRCNN (super-resolution deep convolutional neural network). 

There are 3 layers to this original model with kernel size 9-1-5. Given a lower resolution image (our blurred image), the first convolutional layer use patch extraction and extracts a set of feature maps, the second layer nonlinearly maps those feature maps to higher resolution patches and the last layer is the reconstruction layer. The paper also proposed alternative versions of the original SRCNN model with an increase in the number of layers, the kernel filter size of the hidden layers or the initial kernel filter size. We decided to use an addition of 2 more modified SRCNN models, one with one additional layer and an increase in the kernel filter size of the first hidden layer (9-3-1-5), and one with 2 additional layers (9-1-1-1-5).

For simplicity, we will name our models in the following way:
* Model 1: SRCNN (9-1-5) (Original model)
* Model 2: SRCNN (9-3-1-5) (1 additional layer, increased kernel filter size)
* Model 3: SRCNN (9-1-1-1-5) (2 additional layers)

<div align="center">
<figure>

<img width="800" alt="image" src="https://user-images.githubusercontent.com/39535587/157801780-67732afb-3c28-4878-8592-63feb44bb51d.png"> 
 
 <em> An overview of the structure of the SRCNN model </em>
</figure>
</div>

## Model Performance
The paper experiments with different hyper-parameter settings to improve performance. We decided to do similarly. The paper suggested that using a smaller learning rate in the last layer is essential for the network to converge, so we decided to use 2 different learning rates: 10<sup>-4</sup> and 5 * 10<sup>-5</sup> to compare the performances.

As we train our models, we will try to prove some claims that were made in the paper:
1. A smaller learning rate in the last layer is important for the network to converge.
2. Deeper structure does not always lead to better results.
3. A larger filter size leads to better results.

Performance overview: Model 1 takes about 18s to run 1 poch, Model 2 takes about 28s to run 1 epoch and Model 3 takes about 22s to run 1 epoch. We can see that the larger the kernel filter size is, the longer it takes to train our model.

### Gaussian Filter Deblurring 
#### Learning Rate = 10<sup>-4</sup>:

<div align="center">
<figure>
 
<img width="500" alt="gaussian_loss_model1" src="https://user-images.githubusercontent.com/39535587/157806808-50ccc722-b3ea-4167-8ae3-153ddebe301a.png">
<img width="500" alt="gaussian_loss_model2" src="https://user-images.githubusercontent.com/39535587/157806889-9841f232-8879-41b1-9893-007543446661.png"> 
<img width="500" alt="gaussian_loss_model3" src="https://user-images.githubusercontent.com/39535587/157806945-4d41604b-6b30-496b-b914-cccae08a3d93.png"> 

</figure>
</div>
 
#### Learning Rate = 5 * 10<sup>-5</sup>:

<div align="center">
<figure>
 
<img width="500" alt="gaussian2_loss_model1" src="https://user-images.githubusercontent.com/39535587/157811236-1518234f-36b9-44fa-b56d-612ab230679d.png">
<img width="500" alt="gaussian2_loss_model2" src="https://user-images.githubusercontent.com/39535587/157811249-978719d1-4040-4150-a383-67ad19e65a01.png"> 
<img width="500" alt="gaussian2_loss_model3" src="https://user-images.githubusercontent.com/39535587/157811256-f4c69fe3-56a2-459d-a6a0-db629f9ac6db.png"> 

</figure>
</div>
 
### Box Filter Deblurring 
#### Learning Rate = 10<sup>-4</sup>:

<div align="center">
<figure>
 
<img width="500" alt="box_filter_loss_model1" src="https://user-images.githubusercontent.com/39535587/157806246-c5490c52-b009-4798-9410-a184f6b00128.png"> 
<img width="500" alt="box_filter_loss_model2" src="https://user-images.githubusercontent.com/39535587/157806701-814a8827-c7d0-44ca-80d0-09828c6c236c.png">
<img width="500" alt="box_filter_loss_model3" src="https://user-images.githubusercontent.com/39535587/157806714-1ba85c6e-607a-49ee-95d8-8d428fad3213.png">

</figure>
</div>
 
#### Learning Rate = 5 * 10<sup>-5</sup>:

<div align="center">
<figure>
 
<img width="500" alt="box2_filter_loss_model1" src="https://user-images.githubusercontent.com/39535587/157810425-00cfc9ca-9543-4fd8-b6fb-36e7e18f7198.png">
<img width="500" alt="box2_filter_loss_model2" src="https://user-images.githubusercontent.com/39535587/157810596-3244f063-6a49-4b4d-a86b-4f16c1625be2.png">
<img width="500" alt="box2_filter_loss_model3" src="https://user-images.githubusercontent.com/39535587/157810687-3e6a2be6-df92-448b-a6e8-38d82956d351.png">

</figure>
</div>

### Motion Filter Deblurring 
#### Learning Rate = 10<sup>-4</sup>:

<div align="center">
<figure>
  
<img width="500" alt="motion_loss_model1" src="https://user-images.githubusercontent.com/39535587/157806878-b1805465-997c-47a6-a2c3-4b4a2c4ac3bb.png"> 
<img width="500" alt="motion_loss_model2" src="https://user-images.githubusercontent.com/39535587/157806971-114eea4f-084a-48a8-ad76-a4f04dfa7cda.png"> 
<img width="500" alt="motion_loss_model3" src="https://user-images.githubusercontent.com/39535587/157806907-93393c42-6066-41c1-aa95-633336590e49.png"> 

</figure>
</div>
 
#### Learning Rate = 5 * 10<sup>-5</sup>:
 
<div align="center">
<figure>
  
<img width="500" alt="motion2_loss_model1" src="https://user-images.githubusercontent.com/39535587/157811396-99f3a6a6-504b-433d-b2f3-09f83713dc59.png">
<img width="500" alt="motion2_loss_model2" src="https://user-images.githubusercontent.com/39535587/157811407-d9defb76-7b83-4eaf-a970-d38e1c3fc2dd.png"> 
<img width="500" alt="motion2_loss_model3" src="https://user-images.githubusercontent.com/39535587/157811441-d6fe57b6-fa67-467e-96e6-0cf84650f6ba.png"> 
 
</figure>
</div>

## Model Evaluation
**Learning Rate evaluation:** Looking at the plots, we will notice that the average and overall validation losses of learning rate 10<sup>-4</sup> is much lower than those of learning rate 5 * 10<sup>-5</sup>. Moreover, the losses on the plots of learning rate 5 * 10<sup>-5</sup> appear to have converged as we increase our epochs, suggesting that we've correctly proved claim (1) that smaller learning rates do affect the convergence of the network.

**Model evaluation:** For the same learning rate, we can look at the plots and see that Model 1 and Model 2 have close overall and average validation losses to each other, with one maybe slightly better than another in one blur filter but not in another, suggesting that they performed well. On the other hand, Model 3 performs worse, with much higher loss compared to the other two. This proves claim (2) that deeper structure does not always lead to better results.

**Blur Filter evaluation:** For the same model and learning rate, the overall loss of Gaussian Deblurring and Box Filter Deblurring are quite close to each other. This suggests that our networks performed well in deblurring both Gaussian blurred images and Box Filter blurred images. However, the overall loss of Motion Deblurring is much higher than the other two, implying that our network structures may not be good enough to deblur Motion Blurring filter yet.

## Output
The outputs are sorted in this order for each section (left to right, downwards):
* Original image
* Grayscaled image
* Blurred image
* Model 1 deblurred image
* Model 2 deblurred image
* Model 3 deblurred image

### Gaussian Deblurring
#### Learning Rate = 10<sup>-4</sup>:
 
<div align="center">
<figure>
 
  <img alt="gaussian_original" src="https://user-images.githubusercontent.com/39535587/157814687-1399ccc0-4245-47fa-9158-1a5bd500d3f0.jpg">
  <img alt="gaussian_greyscalel" src="https://user-images.githubusercontent.com/39535587/157814909-f4286106-7a24-4f7e-9f0f-712ee10933ab.jpg">
  <img alt="gaussian_blurred" src="https://user-images.githubusercontent.com/39535587/157814855-dac1a1c8-d395-4240-ab95-0dfb1962c8d7.jpg">
 
</figure>
 
</div>

<div align="center">
<figure>

   <img alt="gaussian_deblurred19_model1" src="https://user-images.githubusercontent.com/39535587/157815393-15bf9e08-8872-4f19-9908-5e0b0517bbaa.jpg">
  <img alt="gaussian_deblurred19_model2" src="https://user-images.githubusercontent.com/39535587/157815403-5a1a1990-405a-40e4-8573-66871289e276.jpg">
  <img alt="gaussian_deblurred19_model3" src="https://user-images.githubusercontent.com/39535587/157815410-402b9482-c9f9-4ff5-9c8c-d3aa1704aef7.jpg">
 
</figure>
 
</div>

#### Learning Rate = 5 * 10<sup>-5</sup>:

<div align="center">
<figure>
 
  <img alt="gaussian2_original" src="https://user-images.githubusercontent.com/39535587/157814782-49854ba6-0368-440d-833d-c148f43b50c4.jpg">
  <img alt="gaussian2_greyscale" src="https://user-images.githubusercontent.com/39535587/157814945-79ff2507-2747-4191-917d-451678058b2f.jpg">
  <img alt="gaussian2_blurred" src="https://user-images.githubusercontent.com/39535587/157815283-e1d03d21-cc07-4a35-b75f-545580a91897.jpg">

</figure>
 
</div>

<div align="center">
<figure>
 
  <img alt="gaussian2_original" src="https://user-images.githubusercontent.com/39535587/157814996-4b656dcf-2847-42c3-961a-ffa1f21f8bcc.jpg">
  <img alt="gaussian2_greyscale" src="https://user-images.githubusercontent.com/39535587/157815237-6ff701be-c55e-44c6-8693-5fdd647610ec.jpg">
  <img alt="gaussian2_blurred" src="https://user-images.githubusercontent.com/39535587/157815246-e457678a-d3be-4058-a805-ed44aa165065.jpg">

</figure>
 
</div>

### Box Filter Deblurring
#### Learning Rate = 10<sup>-4</sup>:

<div align="center">
<figure>
 
  <img alt="box_filter_original" src="https://user-images.githubusercontent.com/39535587/157812715-19b7bca6-ab07-4fa0-ae4c-4795c2657b08.jpg">
  <img alt="box_filter_greyscaled" src="https://user-images.githubusercontent.com/39535587/157812466-1dca112a-eaa7-4e1f-bfd1-a11f9ee6b558.jpg">
  <img alt="box_filter_blurred" src="https://user-images.githubusercontent.com/39535587/157812327-5beb3374-8fab-4ed9-bf66-ed7de6dae4b0.jpg">

</figure>
 
</div>


<div align="center">
<figure>
  
  <img alt="box_filter_deblurred19_model1" src="https://user-images.githubusercontent.com/39535587/157813350-3f3a273e-8726-445c-85cb-e8cbc008b6b7.jpg">
  <img alt="box_filter_deblurred19_model2" src="https://user-images.githubusercontent.com/39535587/157813503-2266e469-7e3f-46af-bc10-e728cf583c83.jpg">
  <img alt="box_filter_deblurred19_model3" src="https://user-images.githubusercontent.com/39535587/157813515-1323c31c-1b2c-4312-b81b-a00c356ff219.jpg">

</figure>
 
</div>

#### Learning Rate = 5 * 10<sup>-5</sup>:

<div align="center">
<figure>
 
  <img alt="box_filter2_original" src="https://user-images.githubusercontent.com/39535587/157814112-0e0fd5e4-18bc-4214-8d04-6899494da706.jpg">
  <img alt="box_filter2_greyscaled" src="https://user-images.githubusercontent.com/39535587/157813795-58d3ce86-bf26-4c39-8cae-1dd423a0249d.jpg">
  <img alt="box_filter2_blurred" src="https://user-images.githubusercontent.com/39535587/157813965-e52175ca-1bc3-474b-b944-fa513cd42b1b.jpg">

</figure>
 
</div>

<div align="center">
 
<figure>

  <img alt="box_filter2_deblurred19_model1" src="https://user-images.githubusercontent.com/39535587/157813837-81cddc32-7279-4757-9c2f-82c056cf5f44.jpg">
  <img alt="box_filter2_deblurred19_model2" src="https://user-images.githubusercontent.com/39535587/157813858-8c608343-055d-474d-b33b-a16fcd2b4913.jpg">
  <img alt="box_filter2_deblurred19_model3" src="https://user-images.githubusercontent.com/39535587/157813870-775a3de9-07d7-4bb9-aa2a-35cf09adedf4.jpg">
 
</figure>
 
</div>


### Motion Deblurring
#### Learning Rate = 10<sup>-4</sup>:

<div align="center">
 
<figure>
 
  <img alt="motion_original" src="https://user-images.githubusercontent.com/39535587/157815918-effc01cd-7ee2-488e-8186-7a9a3d25ace9.jpg">
  <img alt="motion_greyscale" src="https://user-images.githubusercontent.com/39535587/157815729-1151db82-f0b1-4530-8cf0-620c2a624bcf.jpg">
  <img alt="motion_blurred" src="https://user-images.githubusercontent.com/39535587/157815803-1c9d7457-ddac-4062-896e-ac84995a0f90.jpg">

</figure>
 
</div>


<div align="center">
 
<figure>
 
  <img alt="motion_deblurred19_model1" src="https://user-images.githubusercontent.com/39535587/157815757-2eb78ad0-09f2-42ae-b598-75412d78dd09.jpg">
  <img alt="motion_deblurred19_model2" src="https://user-images.githubusercontent.com/39535587/157815779-b5f0ce6b-26d3-45ed-8130-2727649f7cbe.jpg">
  <img alt="motion_deblurred19_model3" src="https://user-images.githubusercontent.com/39535587/157815748-0fea1162-ed69-45b7-9fb5-52c8f3b0fe2a.jpg">

</figure>
 
</div>

#### Learning Rate = 5 * 10<sup>-5</sup>:

<div align="center">
<figure>
 
  <img alt="motion2_original" src="https://user-images.githubusercontent.com/39535587/157815976-70f548c6-a339-4ec4-8e30-885b8c21f978.jpg">
  <img alt="motion2_greyscale" src="https://user-images.githubusercontent.com/39535587/157815605-97d310a8-0e2c-49c0-9aa2-9aa3075b1af8.jpg">
  <img alt="motion2_blurred" src="https://user-images.githubusercontent.com/39535587/157815738-66cfbf36-2560-4bd9-9f66-f99de63e10c2.jpg">
 
</figure>

</div>

<div align="center">
<figure>
 
  <img alt="motion2_deblurred19_model1" src="https://user-images.githubusercontent.com/39535587/157815679-1aea413c-cc85-4dba-a8e9-8ce5689a2fa8.jpg">
  <img alt="motion2_deblurred19_model2" src="https://user-images.githubusercontent.com/39535587/157815663-00b476f4-24d2-4f25-a1a9-5424fdca9066.jpg">
  <img alt="motion2_deblurred19_model3" src="https://user-images.githubusercontent.com/39535587/157815658-d1e31c48-fd10-4adb-b97a-11e27da8544a.jpg">

</figure>

</div>

## Output Comparison
**Learning Rate comparison:** With different learning rates, a deblurring model may have significant differences in terms of output. For example, with learning rate 10<sup>-4</sup>, Model 3's output is significantly worse than that of learning rate 5 * 10<sup>-5</sup>. We can see that the former is very blurry and does not have much difference from the blurred image, while the latter achieves a significant amount of deblurring as compared to the blurred image. This once again proves that learning rate affects models' performances.

**Model comparison:** Overall, the outputs of Model 1 and Model 2 are quite similar, but Model 2's results are slightly better if we look closely. However, Model 3's results are not as good as the others, especially for Gaussian Deblurring and Motion Deblurring. The outputs strengthen the claim that a model's depth is not a guarantee of better results. One more thing to notice is that, the output of Model 2 in Motion Deblurring is significantly better than the other two, as it looks much clearlt and closer to the grayscaled original image, despite not having much lower loss than the others. These outputs help us to conclude that claim (3) is also correct: A larger kernel filter size leads to better results.

**Blur Filter Comparison:** Box Filter Deblurring achieves the most satisfactory outputs, with most models result in significantly deblurred and close to grayscaled original images. The outputs also seems to work well with Gaussian Deblurring although it is essential to do hyper-parameter tuning to achieve good results. However, the outputs of Motion Deblurring are not too well, as they are still very blurry. Even Model 2's results are still blurry, they are just much less compared to the other two. This suggests that our models are not the suitable ones to deblur Motion blurred images yet, and we may need to consider different approaches, such as Autoencoders, GANs and much more complex deep convolutional neural networks should we continue extending our project.

## Conclusion and Future Works
During this project, we have tried a deep learning approach in image deblurring. We have shown that different convolutional neural network models with different depths, kernel sizes and learning rates can achieve different results. We also learned that our models only work for some certain types of blurring filter and still need improvements on more complex blurring filters. We have also successfully proved a number of claims in the paper. 

Based on our current results, as we continue working on this project, we can further extend it by using much more complicated structures to achieve better results, given that our models have a very lightweight and robust structure. Moreover, given limited time, we were only able to train our models on a very small portion of our full dataset, so as we continue our work, we will train our model on the whole dataset to see how they actually perform on very big datasets. Additional performances and results can also be gained by exploring more filters and training methods.

## Related Works
Image Super-Resolution/Image Deblurring is a classical problem in Computer Vision. Therefore, there has been lots of research done to propose new methods of deblurring images with greater outputs. Sun *et al.*'s [paper<sup>[2]</sup>](https://arxiv.org/abs/1503.00593) in 2015 uses a deep convolutional neural network that performs the task of motion blur removal. Ledig *et al.*'s [work<sup>[3]</sup>](https://arxiv.org/abs/1609.04802) involves using a Generative Adversarial Network (GAN) in recovering the finer texture details when we super-resolve at large upscaling factors. Albluwi *et al.* suggested a modified SRCNN model that can work on colored image in their [publishment<sup>[4]</sup>](https://www.researchgate.net/publication/328985265_Image_Deblurring_and_Super-Resolution_Using_Deep_Convolutional_Neural_Networks) recently. Kupyn *et al.* proposed a structure called [DeblurGAN<sup>[5]</sup>](https://arxiv.org/abs/1711.07064) that used Conditional Adversarial Networks to solve the problem of motion deblurring.

## References
1. C. Dong, C. C. Loy, K. He and X. Tang, "Image Super-Resolution Using Deep Convolutional Networks," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, no. 2, pp. 295-307, 1 Feb. 2016, doi: 10.1109/TPAMI.2015.2439281.
2. Jian Sun, Wenfei Cao, Zongben Xu, Jean Ponce. Learning a convolutional neural network for non-uniform motion blur removal. CVPR 2015 - IEEE Conference on Computer Vision and Pattern Recognition 2015, Jun 2015, Boston, United States. IEEE, 2015,.
3. Ledig, Christian & Theis, Lucas & Huszar, Ferenc & Caballero, Jose & Cunningham, Andrew & Acosta, Alejandro & Aitken, Andrew & Tejani, Alykhan & Totz, Johannes & Wang, Zehan & Shi, Wenzhe. (2017). Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. 105-114. 10.1109/CVPR.2017.19. 
4. Albluwi, Fatma & Krylov, Vladimir A. & Dahyot, Rozenn. (2018). Image Deblurring and Super-Resolution Using Deep Convolutional Neural Networks. 1-6. 10.1109/MLSP.2018.8516983. 
5. Kupyn, Orest & Budzan, Volodymyr & Mykhailych, Mykola & Mishkin, Dmytro & Matas, Jiri. (2017). DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks. 
