import os
from sympy import im
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import xmltodict

class YOLODataset(Dataset):
    # 初始化方法
    def __init__(self, image_folder, label_folder, txt_folder, transform, label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.txt_folder = txt_folder
        self.transform = transform
        self.label_transform = label_transform
        self.image_names = os.listdir(self.image_folder)
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_name = self.image_names[index]
        print( "图片名称：" + img_name + "\n")
        img_path = os.path.join(self.image_folder, img_name)  # 路径拼接
        image = Image.open(img_path).convert("RGB") # 强制转换为RGB三通道格式, 即jpg；png为四通道
        w, h = image.size  # 图像的高和宽
        xml_name = img_name.split(".")[0] + ".xml"
        xml_path = os.path.join(self.label_folder, xml_name)
        targets = self._parse_voc_xml(xml_path)
        res = []
        for target in targets:
            x_center = (target[1] + target[3]) / 2 / w
            y_center = (target[2] + target[4]) / 2 / h
            width = (target[3] - target[1]) / w
            height = (target[4] - target[2]) / h
            target[1] = x_center
            target[2] = y_center
            target[3] = width
            target[4] = height
            
            res.append(" ".join([str(a) for a in target]))
        # 写入txt文件
        txt_name = img_name.split(".")[0] + ".txt"
        txt_file = os.path.join(self.txt_folder,txt_name)
        
        with open(txt_file, 'w+') as f:
            f.write('\n'.join(res))    
        targets = torch.tensor(targets)  # 支持：Python列表、NumPy数组、标量值  tensor数据结构更加方便
        
        if self.transform:   # 等同于 self.transform is not None:
            image = self.transform(image)
        return image, targets
    
    def _parse_voc_xml(self, xml_path):
        with open(xml_path, "r", encoding = "utf-8") as f:
            xml_content = f.read()
        # 解析时强制将object转为列表（避免单对象时被自动转字典）
        xml_dict = xmltodict.parse(xml_content, force_list=('object',))
        objects = xml_dict['annotation']['object']
        targets = []
        for object in objects:
            object_name = object['name']
            object_class_id = self._class_name_to_id(object_name)
            object_xmin = float(object['bndbox']['xmin'])
            object_ymin = float(object['bndbox']['ymin'])
            object_xmax = float(object['bndbox']['xmax'])
            object_ymax = float(object['bndbox']['ymax'])
            targets.append([object_class_id, object_xmin, object_ymin, object_xmax, object_ymax])
        return targets
    
    def _class_name_to_id(self, name):
        self.class_list = ["DP", "MF", "BPD", "RING", "Other", 
                           "EM", "BPL", "CM", "EMB", "OK"]
        return self.class_list.index(name)
        
if __name__ == '__main__':
    image_folder = r'D:\Learning\train-V7_20250716160101\Images'   # 图片路径
    label_folder = r'D:\Learning\train-V7_20250716160101\Annotations' # xml文件路径
    txt_folder = r'D:\Learning\train-V7_20250716160101\labels'     # 生成的txt文件路径
    
    if not os.path.exists(txt_folder):
            os.makedirs(txt_folder, exist_ok=True)

    train_dataset = YOLODataset(image_folder, label_folder, txt_folder,
                                transforms.Compose([transforms.ToTensor(),]), None)
    print(len(train_dataset))
    print(train_dataset[0])


