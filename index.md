# Anime Faces Deblurring using Deep Convolutional Neural Network
### CSE 455 Final Project by Minh Hoang and Olivia Mirascian
Published on March 10, 2022

## Abstract
In this project, we explore image deblurring techniques and learning rates to unblur differently blurred animated images. To blur the images, we used three techniques: gaussian filter, box filter, and a custom made motion blurring filter. We then used three different convolutional neural network models and two different learning rates to train and compare the resulting images and loss. We discuss more of the takeaways of this project in this video summary here (make link).

## Related Work


## Problem and Motivation
Clear images are important to capture important moments or to record information, although sometimes the images we capture or download result in being blurrier than we imagined. So for our project, we decided to focus on deblurring images that have been blurred in different ways and identify which models and learning rates are most suitable for different types of blurred images. 

## Dataset
We used a kaggle dataset consisting of 92,000 256x256 images of anime character faces without any metadata. We decided to reduce down the size and use 1,000 images from that dataset in our project. We also learned that the models used for deblurring images currently don't work well with colored images so we decided to apply grayscale filter to our images before blurring and training. The dataset is available here (insert link). The dataset doesn't have any specific intended use but we decided to use it since it was consistent in size and content allowing us to extract important information from our deblurring technique. 

## The SRCNN Model
For our model, we used a deep convolutional neural network called SRCNN (super-resolution deep convolutional neural network). 
There are 3 layers to this original model with kernel size 9-1-5. Given a lower resolution image (our blurred image), the first convolutional layer use patch extraction and extracts a set of feature maps, the second layer nonlinearly maps those feature maps to higher resolution patches and the last layer is the reconstruction layer.

<div align="center">
<figure>

<img width="800" alt="image" src="https://user-images.githubusercontent.com/39535587/157801780-67732afb-3c28-4878-8592-63feb44bb51d.png"> 

</figure>
</div>

The paper experiments with different hyper-parameter settings to improve performance. We decided to do similarly and experiment with different depths, kernel size, and learning rates.

## Performance Plots
For our three models, model 1 was our original model (kernel size 9-1-5), model 2 consisted of a larger kernel and 1 additional layer (9-3-1-5) and model 3 had a smaller kernel with 2 additional layers (9-1-1-1-5)

### Gaussian Filter Deblurring 
#### Learning Rate = 10<sup>-4<sup>

<div align="center">
<figure>
 
<img width="500" alt="gaussian_loss_model1" src="https://user-images.githubusercontent.com/39535587/157806808-50ccc722-b3ea-4167-8ae3-153ddebe301a.png">
<img width="500" alt="gaussian_loss_model2" src="https://user-images.githubusercontent.com/39535587/157806889-9841f232-8879-41b1-9893-007543446661.png"> 
<img width="500" alt="gaussian_loss_model3" src="https://user-images.githubusercontent.com/39535587/157806945-4d41604b-6b30-496b-b914-cccae08a3d93.png"> 

</figure>
</div>
 
#### Learning Rate = 5 * 10<sup>-5<sup>

<div align="center">
<figure>
 
<img width="500" alt="gaussian2_loss_model1" src="https://user-images.githubusercontent.com/39535587/157811236-1518234f-36b9-44fa-b56d-612ab230679d.png">
<img width="500" alt="gaussian2_loss_model2" src="https://user-images.githubusercontent.com/39535587/157811249-978719d1-4040-4150-a383-67ad19e65a01.png"> 
<img width="500" alt="gaussian2_loss_model3" src="https://user-images.githubusercontent.com/39535587/157811256-f4c69fe3-56a2-459d-a6a0-db629f9ac6db.png"> 

</figure>
</div>
 
### Box Filter Deblurring 
#### Learning Rate = 10<sup>-4<sup>

<div align="center">
<figure>
 
<img width="500" alt="box_filter_loss_model1" src="https://user-images.githubusercontent.com/39535587/157806246-c5490c52-b009-4798-9410-a184f6b00128.png"> 
<img width="500" alt="box_filter_loss_model2" src="https://user-images.githubusercontent.com/39535587/157806701-814a8827-c7d0-44ca-80d0-09828c6c236c.png">
<img width="500" alt="box_filter_loss_model3" src="https://user-images.githubusercontent.com/39535587/157806714-1ba85c6e-607a-49ee-95d8-8d428fad3213.png">

</figure>
</div>
 
#### Learning Rate = 5 * 10<sup>-5<sup>

<div align="center">
<figure>
 
<img width="500" alt="box2_filter_loss_model1" src="https://user-images.githubusercontent.com/39535587/157810425-00cfc9ca-9543-4fd8-b6fb-36e7e18f7198.png">
<img width="500" alt="box2_filter_loss_model2" src="https://user-images.githubusercontent.com/39535587/157810596-3244f063-6a49-4b4d-a86b-4f16c1625be2.png">
<img width="500" alt="box2_filter_loss_model3" src="https://user-images.githubusercontent.com/39535587/157810687-3e6a2be6-df92-448b-a6e8-38d82956d351.png">

</figure>
</div>

### Motion Filter Deblurring 
#### Learning Rate = 10<sup>-4<sup>

<div align="center">
<figure>
  
<img width="500" alt="motion_loss_model1" src="https://user-images.githubusercontent.com/39535587/157806878-b1805465-997c-47a6-a2c3-4b4a2c4ac3bb.png"> 
<img width="500" alt="motion_loss_model2" src="https://user-images.githubusercontent.com/39535587/157806971-114eea4f-084a-48a8-ad76-a4f04dfa7cda.png"> 
<img width="500" alt="motion_loss_model3" src="https://user-images.githubusercontent.com/39535587/157806907-93393c42-6066-41c1-aa95-633336590e49.png"> 

</figure>
</div>
 
#### Learning Rate = 5 * 10<sup>-5<sup> 
 
<div align="center">
<figure>
  
<img width="500" alt="motion2_loss_model1" src="https://user-images.githubusercontent.com/39535587/157811396-99f3a6a6-504b-433d-b2f3-09f83713dc59.png">
<img width="500" alt="motion2_loss_model2" src="https://user-images.githubusercontent.com/39535587/157811407-d9defb76-7b83-4eaf-a970-d38e1c3fc2dd.png"> 
<img width="500" alt="motion2_loss_model3" src="https://user-images.githubusercontent.com/39535587/157811441-d6fe57b6-fa67-467e-96e6-0cf84650f6ba.png"> 
 
</figure>
</div>

## Gaussian Blur Results
#### Learning Rate = 10^-4
 
<div align="center">
 
<figure>
 
  <img alt="gaussian_original" src="https://user-images.githubusercontent.com/39535587/157814687-1399ccc0-4245-47fa-9158-1a5bd500d3f0.jpg">
  <img alt="gaussian_greyscalel" src="https://user-images.githubusercontent.com/39535587/157814909-f4286106-7a24-4f7e-9f0f-712ee10933ab.jpg">
  <img alt="gaussian_blurred" src="https://user-images.githubusercontent.com/39535587/157814855-dac1a1c8-d395-4240-ab95-0dfb1962c8d7.jpg">
  
 
</figure>
 
 <figure>
  
  Original, Greyscaled, Blurred
  
  </figure>
 
</div>

<div align="center">
<figure>

   <img alt="gaussian_deblurred19_model1" src="https://user-images.githubusercontent.com/39535587/157815393-15bf9e08-8872-4f19-9908-5e0b0517bbaa.jpg">
  <img alt="gaussian_deblurred19_model2" src="https://user-images.githubusercontent.com/39535587/157815403-5a1a1990-405a-40e4-8573-66871289e276.jpg">
  <img alt="gaussian_deblurred19_model3" src="https://user-images.githubusercontent.com/39535587/157815410-402b9482-c9f9-4ff5-9c8c-d3aa1704aef7.jpg">

 
</figure>

 Model 1, Model 2, Model 3
 
</div>

#### Learning Rate = 5 * 10^-5

<div align="center">
<figure>
 
  <img alt="gaussian2_original" src="https://user-images.githubusercontent.com/39535587/157814782-49854ba6-0368-440d-833d-c148f43b50c4.jpg">
  <img alt="gaussian2_greyscale" src="https://user-images.githubusercontent.com/39535587/157814945-79ff2507-2747-4191-917d-451678058b2f.jpg">
  <img alt="gaussian2_blurred" src="https://user-images.githubusercontent.com/39535587/157815283-e1d03d21-cc07-4a35-b75f-545580a91897.jpg">

</figure>
 
 Original, Greyscaled, Blurred
 
</div>

<div align="center">
<figure>
 
  <img alt="gaussian2_original" src="https://user-images.githubusercontent.com/39535587/157814996-4b656dcf-2847-42c3-961a-ffa1f21f8bcc.jpg">
  <img alt="gaussian2_greyscale" src="https://user-images.githubusercontent.com/39535587/157815237-6ff701be-c55e-44c6-8693-5fdd647610ec.jpg">
  <img alt="gaussian2_blurred" src="https://user-images.githubusercontent.com/39535587/157815246-e457678a-d3be-4058-a805-ed44aa165065.jpg">

</figure>
 
 Model 1, Model 2, Model 3
 
</div>

## Box Deblur Results
#### Learning Rate = 10^-4

<div align="center">
<figure>
 
  <img alt="box_filter_original" src="https://user-images.githubusercontent.com/39535587/157812715-19b7bca6-ab07-4fa0-ae4c-4795c2657b08.jpg">
  <img alt="box_filter_greyscaled" src="https://user-images.githubusercontent.com/39535587/157812466-1dca112a-eaa7-4e1f-bfd1-a11f9ee6b558.jpg">
  <img alt="box_filter_blurred" src="https://user-images.githubusercontent.com/39535587/157812327-5beb3374-8fab-4ed9-bf66-ed7de6dae4b0.jpg">

</figure>
 
 Original, Greyscaled, Blurred
 
</div>


<div align="center">
<figure>
  
  <img alt="box_filter_deblurred19_model1" src="https://user-images.githubusercontent.com/39535587/157813350-3f3a273e-8726-445c-85cb-e8cbc008b6b7.jpg">
  <img alt="box_filter_deblurred19_model2" src="https://user-images.githubusercontent.com/39535587/157813503-2266e469-7e3f-46af-bc10-e728cf583c83.jpg">
  <img alt="box_filter_deblurred19_model3" src="https://user-images.githubusercontent.com/39535587/157813515-1323c31c-1b2c-4312-b81b-a00c356ff219.jpg">

</figure>
 
 Model 1, Model 2, Model 3
 
</div>

#### Learning Rate = 5 * 10^-5

<div align="center">
<figure>
 
  <img alt="box_filter2_original" src="https://user-images.githubusercontent.com/39535587/157814112-0e0fd5e4-18bc-4214-8d04-6899494da706.jpg">
  <img alt="box_filter2_greyscaled" src="https://user-images.githubusercontent.com/39535587/157813795-58d3ce86-bf26-4c39-8cae-1dd423a0249d.jpg">
  <img alt="box_filter2_blurred" src="https://user-images.githubusercontent.com/39535587/157813965-e52175ca-1bc3-474b-b944-fa513cd42b1b.jpg">

</figure>
 
 Original, Greyscaled, Blurred
 
</div>

<div align="center">
 
<figure>

  <img alt="box_filter2_deblurred19_model1" src="https://user-images.githubusercontent.com/39535587/157813837-81cddc32-7279-4757-9c2f-82c056cf5f44.jpg">
  <img alt="box_filter2_deblurred19_model2" src="https://user-images.githubusercontent.com/39535587/157813858-8c608343-055d-474d-b33b-a16fcd2b4913.jpg">
  <img alt="box_filter2_deblurred19_model3" src="https://user-images.githubusercontent.com/39535587/157813870-775a3de9-07d7-4bb9-aa2a-35cf09adedf4.jpg">
 
</figure>
 
 Model 1, Model 2, Model 3
 
</div>


## Motion Deblur Results
#### Learning Rate = 10^-4

<div align="center">
 
<figure>
 
  <img alt="motion_original" src="https://user-images.githubusercontent.com/39535587/157815918-effc01cd-7ee2-488e-8186-7a9a3d25ace9.jpg">
  <img alt="motion_greyscale" src="https://user-images.githubusercontent.com/39535587/157815729-1151db82-f0b1-4530-8cf0-620c2a624bcf.jpg">
  <img alt="motion_blurred" src="https://user-images.githubusercontent.com/39535587/157815803-1c9d7457-ddac-4062-896e-ac84995a0f90.jpg">

</figure>
 
 Original, Greyscaled, Blurred
 
</div>


<div align="center">
 
<figure>
 
  <img alt="motion_deblurred19_model1" src="https://user-images.githubusercontent.com/39535587/157815757-2eb78ad0-09f2-42ae-b598-75412d78dd09.jpg">
  <img alt="motion_deblurred19_model2" src="https://user-images.githubusercontent.com/39535587/157815779-b5f0ce6b-26d3-45ed-8130-2727649f7cbe.jpg">
  <img alt="motion_deblurred19_model3" src="https://user-images.githubusercontent.com/39535587/157815748-0fea1162-ed69-45b7-9fb5-52c8f3b0fe2a.jpg">

</figure>
 
 Model 1, Model 2, Model 3
 
</div>

#### Learning Rate = 5 * 10^-5

<div align="center">
<figure>
 
  <img alt="motion2_original" src="https://user-images.githubusercontent.com/39535587/157815976-70f548c6-a339-4ec4-8e30-885b8c21f978.jpg">
  <img alt="motion2_greyscale" src="https://user-images.githubusercontent.com/39535587/157815605-97d310a8-0e2c-49c0-9aa2-9aa3075b1af8.jpg">
  <img alt="motion2_blurred" src="https://user-images.githubusercontent.com/39535587/157815738-66cfbf36-2560-4bd9-9f66-f99de63e10c2.jpg">
 
</figure>
 
 Original, Greyscaled, Blurred
 
</div>

<div align="center">
<figure>
 
  <img alt="motion2_deblurred19_model1" src="https://user-images.githubusercontent.com/39535587/157815679-1aea413c-cc85-4dba-a8e9-8ce5689a2fa8.jpg">
  <img alt="motion2_deblurred19_model2" src="https://user-images.githubusercontent.com/39535587/157815663-00b476f4-24d2-4f25-a1a9-5424fdca9066.jpg">
  <img alt="motion2_deblurred19_model3" src="https://user-images.githubusercontent.com/39535587/157815658-d1e31c48-fd10-4adb-b97a-11e27da8544a.jpg">

</figure>
 
Model 1, Model 2, Model 3
 
</div>

## Conclusion

 
## References
1. Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. (2014) Learning a Deep Convolutional Network for Image Super-Resolution http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf

2. 
