import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ConcludeAccuracy import *
from waterBodyDataSet import WaterBodyDataSet


def reload_net(path):
    trainednet = torch.load(path,map_location=torch.device('cpu'))
    return trainednet

#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 改变每个轴对应的数值
#     plt.show()


def test(model,testloader):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images,nrow=5))  # nrow是每行显示的图片数量，缺省值为8
    # %5s中的5表示占位5个字符
    # print('GroundTruth: ' , " ".join('%5s' % objectclass.class_names[labels[j]] for j in range(25)))  # 打印前25个GT（test集里图片的标签）
    outputs = model(Variable(images))
    print(outputs)
    print(labels)
    _, predicted = torch.max(outputs.data, 1)
    # 获取分类的预测分数
    sum_score = outputs.data.numpy()
    row, col = sum_score.shape
    score = []
    for i in range(row):
        score.append(np.max(sum_score[i], axis=0))
    true_label = labels.tolist()  # 张量转换为列表
    # print('Predicted: ', " ".join('%5s' % objectclass.class_names[predicted[j]] for j in range(25)))

    pre_value = predicted.tolist()
    score_result = concludescore(pre_value, true_label)
    print('准确率 精确率 召回率：\n', score_result)

    # 绘制ROC曲线
    drawAUC_TwoClass(true_label, score)

    # 绘制混淆矩阵
    cm = confusion_matrix(true_label, pre_value)
    print('混淆矩阵：\n', cm)
    labels_name = ['natural', 'artifial']
    plot_confusion_matrix(cm, labels_name, "HAR Confusion Matrix")


# 打印前25个预测值


if __name__ == "__main__":

    model_path=os.path.join('model', '300net.pkl')
    model = reload_net(model_path)


    test_dir = os.path.join("data", "test.txt")
    dataset = WaterBodyDataSet(test_dir)
    testloader = DataLoader(dataset, batch_size=2)

    test(model,testloader)