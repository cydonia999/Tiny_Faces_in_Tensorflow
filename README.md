# Tiny Face Detector in TensorFlow

 A TensorFlow port(inference only) of Tiny Face Detector from [authors' MatConvNet codes](https://github.com/peiyunh/tiny)[1].

# Requirements

Codes are written in Python. At first install [Anaconda](https://docs.anaconda.com/anaconda/install.html).
Then install [OpenCV](https://github.com/opencv/opencv), [TensorFlow](https://www.tensorflow.org/).

# Usage

## Converting a pretrained model

`matconvnet_hr101_to_pickle` reads weights of the MatConvNet pretrained model and 
write back to a pickle file which is used in a TensorFlow model as initial weights.

1. Download a [ResNet101-based pretrained model(hr_res101.mat)](https://www.cs.cmu.edu/%7Epeiyunh/tiny/hr_res101.mat) 
from the authors' repo.

2. Convert the model to a pickle file by:
```
python matconvnet_hr101_to_pickle.py 
        --matlab_model_path /path/to/pretrained_model 
        --weight_file_path  /path/to/pickle_file
```

## Tesing Tiny Face Detector in TensorFlow

1. Prepare images in a directory. 

2. `tiny_face_eval.py` reads images one by one from the image directory and 
write images to an output directory with bounding boxes of detected faces.
```
python tiny_face_eval.py
  --weight_file_path /path/to/pickle_file
  --data_dir /path/to/input_image_directory
  --output_dir /path/to/output_directory
```

# Neural network diagram

[This](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/networks/ResNet101.pdf)(pdf) is 
a network diagram of the ResNet101-based model used here for an input image(height: 1150, width: 2048, channel: 3).


# Examples

Though this model is developed to detect tiny faces, I apply this to several types of images including 'faces' 
as experiments.

### selfie with many people
This is the same image as one in [the authors' repo](https://github.com/peiyunh/tiny)[1].

![selfie](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/images/selfie.jpg?raw=true)

[Original image](https://github.com/peiyunh/tiny/blob/master/data/demo/selfie.jpg)

### selfie of celebrities
![selfie](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/images/celeb.jpg?raw=true)

[Original image](https://twitter.com/thesimpsons/status/441000198995582976)

### selfie of "celebrities"
Homer and "Meryl Streep" are missed.

![selfie](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/images/celeb2.jpg?raw=true)

[Original image](https://twitter.com/thesimpsons/status/441000198995582976)

### zombies
![selfie](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/images/zombies.jpg?raw=true)

[Original image](http://www.talkingwalkingdead.com/2012/03/walk-on-by.html)

### monkeys
![selfie](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/images/monkeys.jpg?raw=true)

[Original image](http://intisari.grid.id/index.php/Techno/Science/Manusia-Saling-Mengenal-Wajah-Simpanse-Saling-Mengenal-Pantat)

### dogs
![selfie](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/images/dogs.jpg?raw=true)

[Original image](http://www.socialitelife.com/photos/sweet-crazy-woman-adopts-1500-dogs-200-cats/some-may-think-shes-barking-mad-but-one-chinese-woman-adopted-1500-stray-dogs)

### cats
![selfie](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/images/cats.png?raw=true)

[Original image](http://kodex.me/clanak/80268/na-ovom-ostrvu-macke-su-najbrojniji-stanovnici)

### figure1 from a paper[2]
![selfie](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/images/fig1.png?raw=true)

### figure8 from a paper[2]. 
Facebook's face detector failed to detect these faces(as of the paper publication date[14 Feb 2016]).

![selfie](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/images/fig8.png?raw=true)

### figure3 from a paper[2]
![selfie](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/images/fig3.png?raw=true)

### figure6 from a paper[2]
![selfie](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow/blob/master/images/fig6.png?raw=true)

# Acknowledgments

- Many python codes are borrowed from [chinakook's MXNet tiny face detector](https://github.com/chinakook/hr101_mxnet)
- parula colormap table is borrowed from [`fake_parula.py`](https://github.com/BIDS/colormap/blob/master/fake_parula.py).

# Disclaimer

Codes are tested only on CPUs, not GPUs.

# References

1. Hu, Peiyun and Ramanan, Deva,
     Finding Tiny Faces,
     The IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017).
     [project page](https://www.cs.cmu.edu/~peiyunh/tiny/), [arXiv](https://arxiv.org/abs/1612.04402)

2. Michael J. Wilber, Vitaly Shmatikov, Serge Belongie,
     Can we still avoid automatic face detection, 2016.
     [arXiv](https://arxiv.org/abs/1602.04504)

