# CIS
CIS (commodity image searcher), which is able to find a commodity by picture, and quickly find the items you want to buy.  
Note: You need to run "pip install -r requirement.txt" first to install correct version packages.  
Note: If you want to run our code without training, you can use our pre-trained model (.h5 files). The files can be download from [here](https://drive.google.com/drive/folders/1b_K6F-vx8AtBOZVgqMPa4xoa3vflH5zl?usp=sharing).
  
If you have any problem, you can see the [hackmd](https://hackmd.io/SfNwxhIDRpOHmpkUeANxUg) to get more information or contact us!!  
Email: 50903chen@gmail.com

## Dataset
fashion_mist   [[ref]](https://github.com/zalandoresearch/fashion-mnist)
- 28 * 28 gray scale images, associated with a label from 10 classes.
- 60000 training images, 10000 testing images.

| Label  | Description |
| ------------- | ------------- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Module
### utils.py - Some function you will use when you go through process of CIS.
fashion_mnist(split = True)
- description: Load fashion_mnist dataset and return it.
- parameter:
  - split: Whether split dataset to train and test or not.
- return: If split == True return x, y, else return x_train, x_test, y_train, y_test
  
check_label()
- description: Check the predict label of fashion_mnist dataset.

img_transform(path)
- description: Transform the physical image to 28 * 28 * 1 for meeting the model input
- parameter:
  - path: The path of the image
- return: Transformed image

### baseline_model.py - The baseline model is contained.
resnet50(height = 224, width = 224, depth = 3, classes = 1, compile_flag = False)
- description: Load the resnet50 model.
- parameter:
  - height: The height of the image.
  - width: The width of the image.
  - depth: The depth of the image. (Ex: If the image style is grayscale, the depth of the image will be 1.)
  - classes: The numbers of label of the classification task.
  - compile_flag: Whether the model will compile or not.
- return: The model.
  
alexnet(height = 224, width = 224, depth = 3, classes = 1, compile_flag = False)
- description: Load the alexnet model.
- parameter:
  - height: The height of the image.
  - width: The width of the image.
  - depth: The depth of the image. (Ex: If the image style is grayscale, the depth of the image will be 1.)
  - classes: The numbers of label of the classification task.
  - compile_flag: Whether the model will compile or not.
- return: The model.

lenet(height = 224, width = 224, depth = 3, classes = 1, compile_flag = False)
- description: Load the lenet model.
- parameter:
  - height: The height of the image.
  - width: The width of the image.
  - depth: The depth of the image. (Ex: If the image style is grayscale, the depth of the image will be 1.)
  - classes: The numbers of label of the classification task.
  - compile_flag: Whether the model will compile or not.
- return: The model.

model(height = 224, width = 224, depth = 3, classes = 1, compile_flag = False)
- description: Load all the model.
- parameter:
  - height: The height of the image.
  - width: The width of the image.
  - depth: The depth of the image. (Ex: If the image style is grayscale, the depth of the image will be 1.)
  - classes: The numbers of label of the classification task.
  - compile_flag: Whether the model will compile or not.
- return: All the model.

### finetune_model.py - The finetune_model model is contained, only the framework of the model is different from the baseline_model.py, so the description of the function is same as it.

### evaluation.py - Some evaluation method will be contained in it.
F1_score(y_test_classes, y_pred_classes, name)
- description: Calculate the weighted F1_score of the model.
- parameter:
  - y_test_classes: The onehot label of the testing images(ground truth).
  - y_pred_classes: The onehot label of the predicted images which is predicted by model from testing images(predicted result).

predict_probability_to_onehot(y_pred)
- description: Change the probability space to onehot vector
- parameter:
  - y_pred: The result of the model predicting from testing images 
- return: The onehot label of y_pred(testing images)

get_top_rank(data, pic, model, n = 20)
- description: Find the similarity between database(dataset) and the physical(testing) image and return top_n images
- parameter:
  - data: The database(dataset).
  - pic: The physical(testing) image.
  - n: Determine top_n.
- return: The top_n images.

show_top_rank(top_n)
- description: Show the top_n images.
- parameter:
  - top_n: Determine top_n.

## Other Files
- requirement.txt - Record the packages version which we need to use.
- images(folder) - Save the physical images, you can use it when you checking the similarity of the predicted result and the ground truth.
- model(folder) - Save the model(model.h5) we mention before (more details is included in baseline_model & finetune_model), you can use it instead of running the whole training process.
- old_files(folder) - Contain some files when we developing our CIS model, however, those files may not include(or not useful) in our final model, so we abandon them.

## Experiment Results
- ### Accuracy  
![image](https://user-images.githubusercontent.com/66252510/173579842-f50f33c6-ae32-4c01-b30a-47f4d593e669.png)
- ### Weighted F1-score
![image](https://user-images.githubusercontent.com/66252510/173487248-1e5cfb21-e93d-4af0-9fe9-23f9bab440f2.png)

