from utils import fashion_mnist, check_label, img_transform
from baseline_model import model as bl_model
from finetune_model import model as ft_model
from evaluation import F1_score, predict_probability_to_onehot, get_top_rank, show_top_rank
from keras.models import load_model
from sklearn.metrics import accuracy_score

check_label()

x_train, x_test, y_train, y_test, classes = fashion_mnist(split = True)

# # ---------- if you have trained the model, you can skip/ignore it ---------- #
# height, width, depth = x_train[0].shape
# resnet50_baseline, alexnet_baseline, lenet_baseline = \
#     bl_model(height = height, width = width, depth = depth, classes = classes, compile_flag = True)
# resnet50, alexnet, lenet = \
#     ft_model(height = height, width = width, depth = depth, classes = classes, compile_flag = True)
# resnet50_baseline.fit(x_train, y_train, validation_split = 0.3, batch_size = 128, epochs = 30, shuffle = True)
# alexnet_baseline.fit(x_train, y_train, validation_split = 0.3, batch_size = 128, epochs = 30, shuffle = True)
# lenet_baseline.fit(x_train, y_train, validation_split = 0.3, batch_size = 128, epochs = 30, shuffle = True)
# resnet50.fit(x_train, y_train, validation_split = 0.3, batch_size = 128, epochs = 30, shuffle = True)
# alexnet.fit(x_train, y_train, validation_split = 0.3, batch_size = 128, epochs = 30, shuffle = True)
# lenet.fit(x_train, y_train, validation_split = 0.3, batch_size = 128, epochs = 30, shuffle = True)
# # --------------------------------------------------------------------------- #

# # ---------- save the model if necessary ---------- #
# resnet50_baseline.save('model/resnet_baseline.h5')
# alexnet_baseline.save('model/alexnet_baseline.h5')
# lenet_baseline.save('model/lenet_baseline.h5')
# resnet50.save('resnet50_model.h5')
# alexnet.save('alexnet_model.h5')
# lenet.save('lenet_model.h5')
# # ------------------------------------------------- #

# ---------- load the model which we saved ---------- #
alexnet_baseline = load_model('model/alexnet_baseline.h5')
resnet50_baseline = load_model('model/resnet50_baseline.h5')
lenet_baseline = load_model('model/lenet_baseline.h5')
alexnet = load_model('model/alexnet_model.h5')
resnet50 = load_model('model/resnet50_model.h5')
lenet = load_model('model/lenet_model.h5')
# --------------------------------------------------- #

# ---------- if you want to check their F1-score, you can run this code! ---------- #
alexnet_baseline_f1score = F1_score(y_test, predict_probability_to_onehot(alexnet_baseline.predict(x_test)), 'alexnet_baseline')
resnet50_baseline_f1score = F1_score(y_test, predict_probability_to_onehot(resnet50_baseline.predict(x_test)), 'resnet50_baseline')
lenet_baseline_f1score = F1_score(y_test, predict_probability_to_onehot(lenet_baseline.predict(x_test)), 'lenet_baseline')
alexnet_f1score = F1_score(y_test, predict_probability_to_onehot(alexnet.predict(x_test)), 'alexnet')
resnet50_f1score = F1_score(y_test, predict_probability_to_onehot(resnet50.predict(x_test)), 'resnet50')
lenet_f1score = F1_score(y_test, predict_probability_to_onehot(lenet.predict(x_test)), 'lenet')
# --------------------------------------------------------------------------------- #

# ---------- if you want to check their accuracy, you can run this code ---------- #
print("alexnet_baseline accuracy:", accuracy_score(y_test, predict_probability_to_onehot(alexnet_baseline.predict(x_test))))
print("resnet50_baseline accuracy:", accuracy_score(y_test, predict_probability_to_onehot(resnet50_baseline.predict(x_test))))
print("lenet_baseline accuracy:", accuracy_score(y_test, predict_probability_to_onehot(lenet_baseline.predict(x_test))))
print("alexnet accuracy:", accuracy_score(y_test, predict_probability_to_onehot(alexnet.predict(x_test))))
print("resnet50 accuracy:", accuracy_score(y_test, predict_probability_to_onehot(resnet50.predict(x_test))))
print("lenet accuracy:", accuracy_score(y_test, predict_probability_to_onehot(lenet.predict(x_test))))
# -------------------------------------------------------------------------------- #

# ---------- if you want to show concrete example, you can run this code ---------- #
pic8 = img_transform('image/8-1.jpg')
pic8_removebg = img_transform('image/8-1.png')
topn = get_top_rank(x_train, pic8, alexnet, n = 50)
show_top_rank(topn)
# --------------------------------------------------------------------------------- #