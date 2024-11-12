from copy import deepcopy
import spacy
from transformers import pipeline
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:,.0f}'.format


######################################################################################################
# NLP-Task-1: filter deep learning based paper METHOD1: filter out all the deep learning based papers
# this includes a LLM filtering th
# requiement:
# python -m spacy download en_core_web_sm
######################################################################################################

# Load spaCy model for tokenizing sentences
nlp = spacy.load("en_core_web_sm")

# Transformer pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification",model="valhalla/distilbart-mnli-12-3" ,device=0)# model="facebook/bart-large-mnli")


def title_contains_keyword(row, keywords_lower):
        # Use abstract if available, otherwise use title
        text_to_search = str(row['Title'])
        text_lower = text_to_search.lower()  # Convert to lowercase for case-insensitive matching
        return any(keyword in text_lower for keyword in keywords_lower)

def title_keybased_search(data):
  # Define the list of keywords to search within the abstract or title if abstract is NaN
    keywords = [
        'neural network', 'artificial neural network', 'machine learning model', 'feedforward neural network',
        'neural net algorithm', 'multilayer perceptron', 'convolutional neural network', 'recurrent neural network',
        'long short-term memory network', 'CNN', 'GRNN', 'RNN', 'LSTM', 'deep learning', 'deep neural networks',
        'computer vision', 'vision model', 'image processing', 'vision algorithms', 'computer graphics and vision',
        'object recognition', 'scene understanding', 'natural language processing', 'text mining',
        'computational linguistics', 'language processing', 'text analytics', 'textual data analysis',
        'text data analysis', 'text analysis', 'speech and language technology', 'language modeling',
        'computational semantics', 'generative artificial intelligence', 'generative AI', 'generative deep learning',
        'generative models', 'transformer models', 'self-attention models', 'transformer architecture',
        'attention-based neural networks', 'transformer networks', 'sequence-to-sequence models',
        'large language model', 'transformer-based model', 'pretrained language model', 'generative language model',
        'foundation model', 'state-of-the-art language model', 'multimodal model', 'multimodal neural network',
        'vision transformer', 'diffusion model', 'generative diffusion model', 'diffusion-based generative model',
        'continuous diffusion model', "deep learning" ,"machine learning"
    ]

    # Convert keywords to lowercase for case-insensitive matching
    keywords_lower = [k.lower() for k in keywords]

    # Define a function to check if any keyword is in the abstract or title (if abstract is NaN)


    # Filter rows in the DataFrame based on the presence of keywords in Abstract or Title
    dl_based_papers = data[data.apply(lambda row: title_contains_keyword(row, keywords_lower), axis=1)]
    non_dl_based_papers = data[~data.apply(lambda row: title_contains_keyword(row, keywords_lower), axis=1)]

    # Display first few rows of the filtered result
    print(dl_based_papers.head())

    return dl_based_papers, non_dl_based_papers



# Function to extract method sentences
def title_classify_deep_l(data, candidate_labels=["deep learning", "machine learning", "not deep learning"]):
    data_copy = data.copy(deep=True)
    data_copy["deep-learning-used"] = "not-used"
    count=0

    for index, row in data.iterrows():
        abstract = row['Title']
        count+=1

        print("_____________________TITLE____________________________________________________:", count)


        doc = nlp(abstract)
        for sent in doc.sents:
            result = classifier(sent.text, candidate_labels)
            print(result)
            if  result["scores"][result["labels"].index("deep learning")] > 0.5 :
                print("deep ", result["scores"][result["labels"].index("deep learning")])
                data_copy.at[index, 'deep-learning-used'] = "used"  # Assign new data to each row
                break

            elif result["scores"][result["labels"].index("machine learning") ]> 0.5 :

                print("machine ", result["scores"][result["labels"].index("machine learning") ])
                data_copy.at[index, 'deep-learning-used'] = "used"  # Assign new data to each row
                break
    return data_copy



# Function to extract method sentences
def title_classify_cv_txt_mng(filtered_data, candidate_labels=["computer vision", "image processing" , "natural language processing", "text mining","computer vision and natural language processing", "others" ]):
    data_copy = filtered_data.copy(deep=True)
    data_copy["computer_vision"] = "not-used"
    data_copy["text_mining"] = "not-used"
    data_copy["computer_vision_and_text_mining"] = "not-used"
    data_copy["others"] = "not-used"


    for index, row in data_copy.iterrows():
        abstract =  row['Title']
        print("=========================================TITLE========================================")


        doc = nlp(abstract)
        abstract_label = []
        for sent in doc.sents:
            result = classifier(sent.text, candidate_labels)
            #print(result)
            print("abstact_label: ", abstract_label)
            
            if  0 in abstract_label and 1 in abstract_label:
                data_copy.at[index, 'computer_vision_and_text_mining'] = "used"

                #print("++++++++++++++++++++both+++++++++++++++++++++++", abstract_label)
                #time.sleep(3)
                break  # Stop processing sentences once both labels are identified for efficiency


            elif result["scores"][result["labels"].index("computer vision")] > 0.5 or result["scores"][result["labels"].index("image processing")] > 0.5 :
                print("cv ", result["scores"][result["labels"].index("computer vision")])
                print("im ", result["scores"][result["labels"].index("image processing")])
                data_copy.at[index, 'computer_vision'] = "used"  # Assign new data to each row
                abstract_label.append(0)
                #break


            elif result["scores"][result["labels"].index("natural language processing") ]> 0.5 or\
                 result["scores"][result["labels"].index("text mining") ]> 0.5:

                print("nlp ", result["scores"][result["labels"].index("natural language processing") ])
                print("txt ", result["scores"][result["labels"].index("text mining") ])
                data_copy.at[index, 'text_mining'] = "used"  # Assign new data to each row
                abstract_label.append(1)
            elif  result["scores"][result["labels"].index("computer vision and natural language processing")] > 0.5 :

                print("cv and txt ", result["scores"][result["labels"].index("computer vision and natural language processing") ])
                data_copy.at[index, 'computer_vision_and_text_mining'] = "used"  # Assign new data to each row
                abstract_label.append(2)
                #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                break
            

            # If both conditions are met, mark the "computer_vision_and_text_mining" column
            

            #if result["scores"][result["labels"].index("natural language processing") ]> 0.5:
            #    data_copy.at[index, 'text_mining'] = "used"  # Assign new data to each row
                #break
        print("abstract_label: ", abstract_label)

        if len(abstract_label)<1:
            #print("??????????????????????????????????????????? :", abstract_label)
            #time.sleep(3)
            data_copy.at[index, 'others'] = "used"
    return data_copy






# Function to extract method sentences
def title_find_method(data, candidate_labels=["method", "result" ,"conclusion", "context"]):
    data_copy = data.copy(deep=True)
    data_copy["method_used"] = "none"

    method =[]
    for index, row in data.iterrows():
        abstract = row['Title']
        method =[]

        print("####################################TITLE#################################")


        doc = nlp(abstract)
        for sent in doc.sents:
            result = classifier(sent.text, candidate_labels)
            #print(result)
            max_score_index = result["scores"].index(max(result["scores"]))
            max_label = result["labels"][max_score_index]
            if  max_label == "method" :
                print("method", result["scores"][result["labels"].index("method")])
                #print("approach", result["scores"][result["labels"].index("approach")])
                method.append(sent.text)

            #if  result["scores"][result["labels"].index("method")] > 0.50: #or result["scores"][result["labels"].index("approach")]> 0.40 :
            #   print("method", result["scores"][result["labels"].index("method")])
                #print("approach", result["scores"][result["labels"].index("approach")])
            #    method.append(sent.text)

                #data_copy.at[index, 'method used'] = str(sent.text)  # Assign new data to each row


                #break
        joined_method =" ".join(method)
        print("method used :" , joined_method)
        data_copy.at[index, 'method_used'] = joined_method


    return data_copy
