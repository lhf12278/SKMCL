## Shared knowledge guidance and modal consistency learning for Visible-Infrared person re-identification

### Usage
- This project is based on AGW[1] ([paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9336268) and [official code](https://github.com/mangye16/Cross-Modal-Re-ID-baseline)) and DDAG[2] ([paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620222.pdf) and [official code](https://github.com/mangye16/DDAG)).

- Usage of this code is free for research purposes only. 

- Our experimental environment: python3.8.10,torch1.9.0

- Training  
(1)Preparing the dataset(SYSU-MM01[3] ([paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_RGB-Infrared_Cross-Modality_Person_ICCV_2017_paper.pdf)) and RegDB[4] ([paper](https://pdfs.semanticscholar.org/6c51/8aabdbba2c073eab6a3bb4120023851e524c.pdf))). And follow AGW[1] to do data preprocessing.  
(2)To begin training.(See the code and our paper for more details)   
```
python train.py
```
- Testing.  
(1)Preparing the dataset.(SYSU-MM01[3] ([paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_RGB-Infrared_Cross-Modality_Person_ICCV_2017_paper.pdf)) and RegDB[4] ([paper](https://pdfs.semanticscholar.org/6c51/8aabdbba2c073eab6a3bb4120023851e524c.pdf))).  
(2)Downloading the parameter files trained in this paper.( Using to verify the effectiveness of the proposed method).[Google Drive](https://drive.google.com/drive/folders/1zW5kJKGDONTv9J-IhB1SKpR1PyCGiZQD?usp=sharing).  
(3)To begin testing.(See the code for more details)    
```
python test.py
```

- If you have any questions in use, please contact me. [xukaixiong@stu.kust.edu.cn](xukaixiong@stu.kust.edu.cn) . 

- Reference
```
[1]Ye M, Shen J, Lin G, et al. Deep learning for person re-identification: A survey and outlook[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.  
[2]Ye M, Shen J, J. Crandall D, et al. Dynamic dual-attentive aggregation learning for visible-infrared person re-identification[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVII 16. Springer International Publishing, 2020: 229-247.
[3]Wu A, Zheng W S, Yu H X, et al. RGB-infrared cross-modality person re-identification[C]//Proceedings of the IEEE international conference on computer vision. 2017: 5380-5389.
[4]Nguyen D T, Hong H G, Kim K W, et al. Person recognition system based on a combination of body images from visible light and thermal cameras[J]. Sensors, 2017, 17(3): 605.
```
