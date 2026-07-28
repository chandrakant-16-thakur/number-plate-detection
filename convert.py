import os
import xml.etree.ElementTree as ET

xml_folder = "archive/annotations"
image_folder = "archive/images"

output = "dataset"

for folder in [
    "train/images","train/labels",
    "valid/images","valid/labels",
    "test/images","test/labels"
]:
    os.makedirs(os.path.join(output, folder), exist_ok=True)

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0

    w = box[1] - box[0]
    h = box[3] - box[2]

    return x*dw, y*dh, w*dw, h*dh

xml_files = os.listdir(xml_folder)
import random
random.shuffle(xml_files)

for i, xml_file in enumerate(xml_files):

    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    filename = root.find("filename").text

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    txt_name = filename.replace(".png",".txt").replace(".jpg",".txt")

    if i < len(xml_files)*0.7:
        img_dir="train/images"
        label_dir="train/labels"
    elif i < len(xml_files)*0.9:
        img_dir="valid/images"
        label_dir="valid/labels"
    else:
        img_dir="test/images"
        label_dir="test/labels"

    import shutil
    shutil.copy(
        os.path.join(image_folder, filename),
        os.path.join(output, img_dir, filename)
    )

    out = open(os.path.join(output, label_dir, txt_name),"w")

    for obj in root.iter("object"):

        xmlbox=obj.find("bndbox")

        xmin=float(xmlbox.find("xmin").text)
        xmax=float(xmlbox.find("xmax").text)
        ymin=float(xmlbox.find("ymin").text)
        ymax=float(xmlbox.find("ymax").text)

        bb=convert((w,h),(xmin,xmax,ymin,ymax))

        out.write("0 "+" ".join([str(a) for a in bb])+"\n")

    out.close()

print("Dataset Ready")