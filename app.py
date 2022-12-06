# save this as app.py
from flask import Flask, render_template
from flask import request
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import spacy
import scispacy
from questions import ques

import warnings
warnings.filterwarnings('ignore')

general_nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_sci_sm")

svm_model = pickle.load(open('model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
kmeans_model = pickle.load(open('km_model.pkl', 'rb'))
# symptoms_model = pickle.load(open('symptoms_diabetes.pkl', 'rb'))

# sentence = nlp('''Hello there, I am Ben Whittaker. My gender is male and I am 55 years old, I have high blood pressure and more cholestrol levels than normal. My BMI is 26, last year I had heart stroke, I have heart condition, my normal health condition is always good, I have never experienced discomfort either physical or mental condition, I have leg pains and some level of difficulty while walking.''')

# for ent in sentence.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)



startQuiz = False

# x_test = np.array([[0,	0,	1,	31,	1,	0,	0,	0,	1,	1,	0,	1,	0,	4,	0,	0,	0,	1,	6,	3]])


app = Flask(__name__)

def check_entities(sentence):
    if (sentence and len(sentence) > 0):
        general_nlp_for_sentence = general_nlp(sentence)
        nlp_for_sentence = nlp(sentence)
        general_entities = []
        entities = []
        for ent in general_nlp_for_sentence.ents:
            print("General entities", ent.text, ent.start_char, ent.end_char, ent.label_)
            general_entities.append(ent.text)
        for ent in nlp_for_sentence.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)
            entities.append(ent.text)
        if (entities and len(entities) > 0):
            return general_entities+entities
        else:
            return []

def getCatergory(entities):
    if (entities and len(entities) > 0):
        keyword = "diabetes"
        if keyword in entities:
            return getReply("start")
        elif (len(entities) > 15):
           print("Entities:", entities)
           get_inputs = make_np_arr(entities)
           print("Inputs from entities : ",get_inputs)
           svm_predicted_value = svm_model.predict(get_inputs)
           km_predicted_value = kmeans_model.predict(get_inputs)
           rf_predicted_value = rf_model.predict(get_inputs)
           all_predictions = {
                "svm_predicted" : (svm_predicted_value == 0 and {"output" : "No Diabetes"} ) or (svm_predicted_value == 1 and {"output" : "Pre Diabetes"} ) or (svm_predicted_value == 2 and {"output" : "Diabetes"} ),
                "km_predicted" : (km_predicted_value == 0 and {"output" : "No Diabetes"} ) or (km_predicted_value == 1 and {"output" : "Pre Diabetes"} ) or (km_predicted_value == 2 and {"output" : "Diabetes"} ),
                "rf_predicted" : (rf_predicted_value == 0 and {"output" : "No Diabetes"} ) or (rf_predicted_value == 1 and {"output" : "Pre Diabetes"} ) or (rf_predicted_value == 2 and {"output" : "Diabetes"} ) }
           print("got output :",all_predictions)
           return {
                "isQuestion" : False,
                 "reply" : "Your predicted stage of diabetes is SVM => "+((svm_predicted_value == 0 and "No Diabetes" )or (svm_predicted_value == 1 and "Pre Diabetes" ) or (svm_predicted_value == 2 and "Diabetes" ))+ " K means => "+((km_predicted_value == 0 and "No Diabetes" ) or (km_predicted_value == 1 and "Pre Diabetes" ) or (km_predicted_value == 2 and "Diabetes"))+" Random Forest => "+((rf_predicted_value == 0 and "No Diabetes" ) or (rf_predicted_value == 1 and "Pre Diabetes" ) or (rf_predicted_value == 2 and "Diabetes" ))
                 }
        else:
            print("Entities:", entities)
            return getReply("Hi")
    else:
        return getReply("hi")

def getAge_from_entities(entities):
    age=[]
    for item in entities:
        years = item.find("years")
        if years != -1:
            return entities[item] and entities[item].split("") and entities[item].split("")[0]
        else :
            return 40 # age not found in entities

def make_np_arr(entities):
    print(entities.index("blood pressure"),'output')
    np_arr = [
        np.where(entities.count("blood pressure") > 0, 1 ,0),
          np.where(entities.count("cholestrol") > 0, 1 ,0),
          np.where(entities.count("BMI") > 0, 25 ,0),
          np.where(entities.count("blood pressure") > 0, 1 ,0),
          np.where(entities.count("heart condition") > 0, 1 ,0),
          np.where(entities.count("health condition") > 0, 1 ,0),
          np.where(entities.count("discomfort") > 0, 20 ,0),
          np.where(entities.count("leg pains") > 0, 1 ,0),
          np.where((entities.count("gender") > 0 and entities[entities.index("gender")+1] == "female"), 1 ,0),
          getAge_from_entities(entities),
    ]
    test_value = np.array([np_arr])
    reshaping_np_arr = np.reshape(test_value, (-1, 10))
    print("Array from entities:",reshaping_np_arr)
    return reshaping_np_arr

def convert_to_npArray(data):
    modified = [
        np.where(data["high_bp"] == "yes", 1 ,0),
          np.where(data["cholestrol"]== "yes", 1 ,0),
        #   np.where(data["cholestrol_level"]== "yes", 1 ,0),
          int(data["bmi"]),
          # np.where(data["smoker"]== "yes", 1 ,0),
          np.where(data["stroke"]== "yes", 1 ,0),
          np.where(data["heart_condition"]== "yes", 1 ,0),
          # np.where(data["physical_activity"]== "yes", 1 ,0),
          # np.where(data["fruits"]== "yes", 1 ,0),
         # np.where( data["veggies"]== "yes", 1 ,0),
          # np.where(data["alcohol_consumption"]== "yes", 1 ,0),
          # np.where(data["doctor_consultation"]== "yes", 1 ,0),
          np.where(data["general_health"]== "good", 1 ,0),
          # int(data["mental_health"]),
          int(data["physical_health"]),
          np.where(data["walk"]== "yes", 1 ,0),
          np.where(data["sex"]== "female", 1 ,0),
          int(data["age"]),
    ]
    test_value = np.array([modified])
    print("After converting given inputs to np array: \n", test_value)
    return test_value

@app.route("/")
def display_chat():
    return render_template('chat.html')

# @app.route('/chat', methods=['GET'])
# def display_chat():
#     return render_template('chat.html')

def getReply(message):
    if message and len(message) > 0:
        if message == "Hi":
            return {
                "reply" : "Great, I am chat bot I can help you with predicting your diabetes stages. Please let me know your symptoms. Thanks",
                "isQuestion" : False
            }
        elif message == "start":
            
            return {
                "reply" : ques()[0],
                "isQuestion" : True
            }
        else:
            return {
                "reply" : "Sorry, didn't get your text.",
                "isQuestion" : False
            }
    else: 
        return {
                "reply" : "Hi! there, how can I help you?",
                "isQuestion" : False
            }

@app.route('/send_message', methods=['POST'])
def send_message():
    if request.method == 'POST':
        data = request.json
        print("received message",data)
        checking_entities = check_entities(data)
        get_text_category = getCatergory(checking_entities)
        # get_reply = getReply(data)
        if get_text_category and len(get_text_category) > 0:
            return {
                "message" : get_text_category
            }
        else:
            message = {
                "message" : "Hi! how are you doing?"
            }
            return message

@app.route('/submit', methods=['POST'])
def welcome():
    if request.method == 'POST':
        data = request.json
        print("received data",data)
        result = convert_to_npArray(data)
        svm_predicted_value = svm_model.predict(result)
        km_predicted_value = kmeans_model.predict(result)
        rf_predicted_value = rf_model.predict(result)
        all_predictions = {
                                "svm_predicted" : (svm_predicted_value == 0 and {"output" : "No Diabetes"} ) or (svm_predicted_value == 1 and {"output" : "Pre Diabetes"} ) or (svm_predicted_value == 2 and {"output" : "Diabetes"} ),
                                "km_predicted" : (km_predicted_value == 0 and {"output" : "No Diabetes"} ) or (km_predicted_value == 1 and {"output" : "Pre Diabetes"} ) or (km_predicted_value == 2 and {"output" : "Diabetes"} ),
                                "rf_predicted" : (rf_predicted_value == 0 and {"output" : "No Diabetes"} ) or (rf_predicted_value == 1 and {"output" : "Pre Diabetes"} ) or (rf_predicted_value == 2 and {"output" : "Diabetes"} ) }
        print("got output :",all_predictions)
        
        return all_predictions
        # (predicted_value == 0 and {"output" : "No Diabetes"} ) or (predicted_value == 1 and {"output" : "Pre Diabetes"} ) or (predicted_value == 1 and {"output" : "Diabetes"} )

# @app.route("/")
# def hello():
#     get_data = pd.read_excel("predict.xlsx")
#     print(get_data.head())
#     encoding_description_column = pd.get_dummies(get_data["Description"])
#     print("values: \n", encoding_description_column.head())
    
#     x_test = encoding_description_column
#     value = symptoms_model.predict(x_test)
#     print("output: \n")
#     return "it will work"

CORS(app)