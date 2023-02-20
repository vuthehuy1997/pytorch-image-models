import os
import json
import shutil
import random

data_path = 'datasets/food-101/images'
train_json = 'datasets/food-101/meta/train.json'
test_json = 'datasets/food-101/meta/test.json'
ratio = 0.2

train_out = 'datasets/food-101/train'
validation_out = 'datasets/food-101/validation'
test_out = 'datasets/food-101/test'

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok = True)
    return


create_folder(train_out)
create_folder(validation_out)
f = open(train_json)
d = json.load(f)
for k,v in d.items():
    random.shuffle(v)

    train_cat = os.path.join(train_out, k)
    create_folder(train_cat)
    val_cat = os.path.join(validation_out, k)
    create_folder(val_cat)

    length = len(v)
    for name in v[:int(length*ratio)]:
        shutil.copy(os.path.join(data_path, name) + '.jpg', os.path.join(validation_out, name) + '.jpg')
    for name in v[int(length*ratio):]:
        shutil.copy(os.path.join(data_path, name) + '.jpg', os.path.join(train_out, name) + '.jpg')


create_folder(test_out)
f = open(test_json)
d = json.load(f)
for k,v in d.items():
    random.shuffle(v)

    test_cat = os.path.join(test_out, k)
    create_folder(test_cat)

    length = len(v)
    for name in v:
        shutil.copy(os.path.join(data_path, name) + '.jpg', os.path.join(test_out, name) + '.jpg')


