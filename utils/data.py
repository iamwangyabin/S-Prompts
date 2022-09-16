import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from utils.datautils.core50.core50data import CORE50


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iGanFake(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = args["class_order"]
        self.class_order = class_order

    def download_data(self):

        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'train')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'val')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)




class iCoRE50(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = args["class_order"]
        self.class_order = class_order

    def download_data(self):
        self.dataset_generator = CORE50(root=data_path, scenario="ni")
        images, labels = self.dataset_generator.get_test_set()
        labels = labels.tolist()


        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'train')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'val')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)



class iCore50(iData):

    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(8*50).tolist()

    def download_data(self):
        datagen = CORE50(root='/home/wangyabin/workspace/datasets/core50/data/core50_128x128', scenario="ni")

        dataset_list = []
        for i, train_batch in enumerate(datagen):
            imglist, labellist = train_batch
            labellist += i*50
            imglist = imglist.astype(np.uint8)
            dataset_list.append([imglist, labellist])
        train_x = np.concatenate(np.array(dataset_list)[:, 0])
        train_y = np.concatenate(np.array(dataset_list)[:, 1])
        self.train_data = train_x
        self.train_targets = train_y

        test_x, test_y = datagen.get_test_set()
        test_x = test_x.astype(np.uint8)
        self.test_data = test_x
        self.test_targets = test_y
        # import pdb;pdb.set_trace()
