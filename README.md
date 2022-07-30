# Manufacturing part recognition by training 2D views

This study is a 3D object recognition by 2D images project for manufacturing parts along an assembly line. We adopted the "Multi-View Convolutional Neural Networks for 3D Shape Recognition" (MVCNN) [[1]](#1) paper and add new approaches for the model to recognize 101 parts (classes). The paper uses 12 (and more) rendered views whereas it is not practical to get several views, i.e., 15-20, for inference in a real-world setting. Thus, we kept the number of views small (3 views). Initially, the model was given fewer classes (first 10, then 20, then 50). However, the model performance dropped as the number of classes increased. Then, we chose to apply two-stage architecture, where we grouped classes randomly, (let's say classes 1, 6, 59, 95, 30 are labeled as group 1; classes 12, 77, 64, 3, 88 are labeled as group 2 and so on). The goal of the first stage is to label the groups of classes correctly whereas, the second stage recognizes the correct class label among each randomly assigned group.

![model architecture](https://user-images.githubusercontent.com/7215154/181904515-1cab3b5f-1b85-4456-82c3-7fdb41319bfa.svg)



### References
<a id="1">[1]</a> 
Su, Hang, et al. "Multi-view convolutional neural networks for 3d shape recognition." Proceedings of the IEEE international conference on computer vision. 2015.
