from __future__ import division, print_function
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import ImageFile
import numpy as np
from glob import glob
import cv2                
import os
import torch
from torchvision import datasets
from torchvision import transforms
import copy
import torch.nn as nn
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH_dog = 'model_transfer.pt'
# Load your trained model
VGG16 = models.vgg16(pretrained=True)
model_transfer=copy.deepcopy(VGG16)

input_lastlayer = model_transfer.classifier[6].in_features
out_lastlayer = 133
lastlayer = nn.Linear(input_lastlayer,out_lastlayer)
model_transfer.classifier[6] = lastlayer
model_transfer.load_state_dict(torch.load(MODEL_PATH_dog,map_location=torch.device('cpu')))
model_transfer.eval()


ImageFile.LOAD_TRUNCATED_IMAGES = True

def VGG16_predict(img_path):
    data_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    im = Image.open(img_path)
    img_data = data_transform(im).unsqueeze_(0)
    with torch.no_grad():
        VGG16.eval()
        output = VGG16(img_data)
    value, index = output[0].max(0)
    return index 

def dog_detector(img_path):
    index = VGG16_predict(img_path)
    if index >= 151 and index <= 268:
        return True
    else:
        return False # true/false



def predict_breed_transfer(img_path):
    data_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    im = Image.open(img_path)
    img_data = data_transform(im).unsqueeze_(0)
    with torch.no_grad():
        model_transfer.eval()
        output = model_transfer(img_data)
    value, index = output[0].max(0)
    return index

def detect_human(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    if(len(faces)>0):
        return True
    else:
        return False
mapping_dict = {0: 'Affenpinscher',
 1: 'Afghan hound',
 2: 'Airedale terrier',
 3: 'Akita',
 4: 'Alaskan malamute',
 5: 'American eskimo dog',
 6: 'American foxhound',
 7: 'American staffordshire terrier',
 8: 'American water spaniel',
 9: 'Anatolian shepherd dog',
 10: 'Australian cattle dog',
 11: 'Australian shepherd',
 12: 'Australian terrier',
 13: 'Basenji',
 14: 'Basset hound',
 15: 'Beagle',
 16: 'Bearded collie',
 17: 'Beauceron',
 18: 'Bedlington terrier',
 19: 'Belgian malinois',
 20: 'Belgian sheepdog',
 21: 'Belgian tervuren',
 22: 'Bernese mountain dog',
 23: 'Bichon frise',
 24: 'Black and tan coonhound',
 25: 'Black russian terrier',
 26: 'Bloodhound',
 27: 'Bluetick coonhound',
 28: 'Border collie',
 29: 'Border terrier',
 30: 'Borzoi',
 31: 'Boston terrier',
 32: 'Bouvier des flandres',
 33: 'Boxer',
 34: 'Boykin spaniel',
 35: 'Briard',
 36: 'Brittany',
 37: 'Brussels griffon',
 38: 'Bull terrier',
 39: 'Bulldog',
 40: 'Bullmastiff',
 41: 'Cairn terrier',
 42: 'Canaan dog',
 43: 'Cane corso',
 44: 'Cardigan welsh corgi',
 45: 'Cavalier king charles spaniel',
 46: 'Chesapeake bay retriever',
 47: 'Chihuahua',
 48: 'Chinese crested',
 49: 'Chinese shar-pei',
 50: 'Chow chow',
 51: 'Clumber spaniel',
 52: 'Cocker spaniel',
 53: 'Collie',
 54: 'Curly-coated retriever',
 55: 'Dachshund',
 56: 'Dalmatian',
 57: 'Dandie dinmont terrier',
 58: 'Doberman pinscher',
 59: 'Dogue de bordeaux',
 60: 'English cocker spaniel',
 61: 'English setter',
 62: 'English springer spaniel',
 63: 'English toy spaniel',
 64: 'Entlebucher mountain dog',
 65: 'Field spaniel',
 66: 'Finnish spitz',
 67: 'Flat-coated retriever',
 68: 'French bulldog',
 69: 'German pinscher',
 70: 'German shepherd dog',
 71: 'German shorthaired pointer',
 72: 'German wirehaired pointer',
 73: 'Giant schnauzer',
 74: 'Glen of imaal terrier',
 75: 'Golden retriever',
 76: 'Gordon setter',
 77: 'Great dane',
 78: 'Great pyrenees',
 79: 'Greater swiss mountain dog',
 80: 'Greyhound',
 81: 'Havanese',
 82: 'Ibizan hound',
 83: 'Icelandic sheepdog',
 84: 'Irish red and white setter',
 85: 'Irish setter',
 86: 'Irish terrier',
 87: 'Irish water spaniel',
 88: 'Irish wolfhound',
 89: 'Italian greyhound',
 90: 'Japanese chin',
 91: 'Keeshond',
 92: 'Kerry blue terrier',
 93: 'Komondor',
 94: 'Kuvasz',
 95: 'Labrador retriever',
 96: 'Lakeland terrier',
 97: 'Leonberger',
 98: 'Lhasa apso',
 99: 'Lowchen',
 100: 'Maltese',
 101: 'Manchester terrier',
 102: 'Mastiff',
 103: 'Miniature schnauzer',
 104: 'Neapolitan mastiff',
 105: 'Newfoundland',
 106: 'Norfolk terrier',
 107: 'Norwegian buhund',
 108: 'Norwegian elkhound',
 109: 'Norwegian lundehund',
 110: 'Norwich terrier',
 111: 'Nova scotia duck tolling retriever',
 112: 'Old english sheepdog',
 113: 'Otterhound',
 114: 'Papillon',
 115: 'Parson russell terrier',
 116: 'Pekingese',
 117: 'Pembroke welsh corgi',
 118: 'Petit basset griffon vendeen',
 119: 'Pharaoh hound',
 120: 'Plott',
 121: 'Pointer',
 122: 'Pomeranian',
 123: 'Poodle',
 124: 'Portuguese water dog',
 125: 'Saint bernard',
 126: 'Silky terrier',
 127: 'Smooth fox terrier',
 128: 'Tibetan mastiff',
 129: 'Welsh springer spaniel',
 130: 'Wirehaired pointing griffon',
 131: 'Xoloitzcuintli',
 132: 'Yorkshire terrier'}

def run_app(img_path,model_transfer,mapping_dict):
    ishuman = detect_human(img_path)
    if ishuman:
        return str("Hooman detected. This classifier is really for doggos but if you were a doggo, you'd be....{}".format(mapping_dict.get(int(predict_breed_transfer(img_path)))))
    elif dog_detector(img_path):
        return str("Hey good boi, My guess is you'd be... {}".format(mapping_dict.get(int(predict_breed_transfer(img_path)))))
    else:
        return str("ERROR : NO HUMAN OR DOGGO DETECTED!")
    return


print('Model loaded. Check http://127.0.0.1:5000/')



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("upload function entered")
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads')
        file_path_filename = os.path.join(
            file_path, secure_filename(f.filename))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        print(file_path_filename)
        f.save(file_path_filename)

        # Make prediction
        preds = run_app(file_path_filename, model_transfer, mapping_dict)
        return preds
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

