import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def F1_score(y_test_classes, y_pred_classes, name):    
    precision = precision_score(y_test_classes, y_pred_classes, average = 'weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average = 'weighted')
    score = f1_score(y_test_classes, y_pred_classes, average = 'weighted')
    print(name + "'s weighted Precision:", precision)
    print(name + "'s weighted Recall:", recall)
    print(name + "'s weighted F1-score:", score)


def predict_probability_to_onehot(y_pred):
    for i in range(len(y_pred)):
        max_value= max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0
    
    return y_pred

def get_top_rank(data, pic, model, n = 20):
    
    pic_pred_prob = model.predict(pic) 
    pic_pred_label = np.array(pic_pred_prob).argmax().astype(int)
    print(f'predict label: {pic_pred_label}')
    
    y_pred = model.predict(data)
    size = data.shape[0]
    probability = np.zeros((size, 2))

    for i, item in enumerate(y_pred):
        probability[i] = (np.array(item).argmax().astype(int), np.array(item.max()))
        condition = np.where(probability[:, 0] == pic_pred_label)
        probability_result = np.concatenate((condition[0].reshape(len(condition[0]), 1), 
                            probability[condition[0]][:,1].reshape(probability[condition[0]][:,1].shape[0], 1)), axis = 1)
        result = probability_result[probability_result[:, 1].argsort()][::-1][:size]
    top_n = data[result[:n, 0].astype(int)]
    
    return top_n

def show_top_rank(top_n):
    for i in range(top_n.shape[0]):
        plt.axis('off')
        plt.imshow(top_n[i], cmap = 'gray')
        plt.show()
