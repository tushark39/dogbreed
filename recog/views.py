from rest_framework.response import Response
from rest_framework.generics import CreateAPIView
from rest_framework import status
from .serializers import ImageSerializer
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib.pyplot import imread
from rest_framework.parsers import MultiPartParser
import numpy as np
import os

# Create your views here.
MODEL_PATH ="/home/tushar/ml_t/dogBreedRecog/recog/dog-breed-full-model.h5"

unique_breed = np.array(['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier'])

class DogBreed(CreateAPIView):
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH,custom_objects={"KerasLayer":hub.KerasLayer})
        if self.model:
            print("[*] Model Loaded")
        else:
            print("[*] failed to load the model")
            return Response({"success": False, "error": "failed to load the model"}, status=status.HTTP_400_BAD_REQUEST)


    parser_classes = (MultiPartParser,)
    def get_serializer_class(self):
        return ImageSerializer
    
    
    
    def save_uploaded_image(self, image):
        save_folder = "/home/tushar/ml_t/dogBreedRecog/recog/images/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, "01.jpg")
        with open(save_path, 'wb') as f:
            f.write(image.read())
        return save_path
    
    def get_image_label(self,prediction):
        return unique_breed[np.argmax(prediction)]
    
    def preprocess_image(self,image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image,channels=3)
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.image.resize(image,size=[224,224])
        return image
    def preprocess_custom_image(self,image_path):
        data = tf.data.Dataset.from_tensor_slices(tf.constant([image_path]))
        data = data.map(self.preprocess_image)
        data_batch = data.batch(32)
        return data_batch
    def predict(self,model,image_path):
        data_batch = self.preprocess_custom_image(image_path)
        pred_prob = model.predict(data_batch)[0]
        
        return pred_prob
    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)

        if serializer.is_valid():
            image = serializer.validated_data['image']
            try:
                path = self.save_uploaded_image(image)
                self.pred_prob = self.predict(self.model,path)
                self.formatted_accuracy = round(np.max(self.pred_prob) * 100, 2)
                self.res = {
                    "breed" : self.get_image_label(self.pred_prob),
                    "accuracy": self.formatted_accuracy
                }
                return Response({"success":self.res}, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({"success": False, "error": f"Error processing image: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response({"success": False, "error": "Invalid data"}, status=status.HTTP_400_BAD_REQUEST)