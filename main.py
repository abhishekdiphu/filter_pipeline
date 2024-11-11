from copy import deepcopy
import spacy
from transformers import pipeline
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:,.0f}'.format

# PREPROCESSING STEP :
# Load a CSV file into a DataFrame

df = pd.read_csv('collection_with_abstracts.csv')

# Display the first few rows
#print(df.head())

# Display the first few rows
print(df.head(2))
print("columns exsisted in the original csv dataset: ", df.columns.tolist())

nan_stats = df.isna().sum()
print("infromation missing for each column: ", nan_stats)


# make a new dataframe , where only PMID Title and Abstract column is present !
df_pmid_title_abstract = df[['PMID', 'Title', 'Abstract']]
df_pmid_title_abstract.isna().sum()
print("length of rows: ", df_pmid_title_abstract.shape[0])
print(df_pmid_title_abstract.describe(include='all'))


#make a new dataframe after removing rows with NaN in the 'Abstract' column
pmid_title_abstract_df_cleaned = df_pmid_title_abstract.dropna(subset=['Abstract'])

# Display the first few rows of the cleaned DataFrame
pmid_title_abstract_df_cleaned.head()
print(pmid_title_abstract_df_cleaned.isna().sum())
print("length of rows: ", pmid_title_abstract_df_cleaned.shape[0])



#  Make a new dataframe withh PMID, Title, Abstract , where NaN is present, which can be used later on for
#  for filtration process seperately  
df_abstract_only_nan = df_pmid_title_abstract[df_pmid_title_abstract['Abstract'].isna()]
print("length of row", df_abstract_only_nan.shape[0])
print(df_abstract_only_nan.isna().sum())
print(df_abstract_only_nan.head())




######################################################################################################
# NLP-Task-1: filter deep learning based paper METHOD1: filter out all the deep learning based papers
######################################################################################################

def contains_keyword(row, keywords_lower):
        # Use abstract if available, otherwise use title
        text_to_search = str(row['Abstract']) if pd.notna(row['Abstract']) else str(row['Title'])
        text_lower = text_to_search.lower()  # Convert to lowercase for case-insensitive matching
        return any(keyword in text_lower for keyword in keywords_lower)

def keybased_search(data):
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
    dl_based_papers = data[data.apply(lambda row: contains_keyword(row, keywords_lower), axis=1)]
    non_dl_based_papers = data[~data.apply(lambda row: contains_keyword(row, keywords_lower), axis=1)]

    # Display first few rows of the filtered result
    print(dl_based_papers.head())

    return dl_based_papers, non_dl_based_papers


# fltration based on Key based filtering (prelinimary filtration process: rule based) 
# run the function above 
df_based_pp, non_dl_based_pp = keybased_search(pmid_title_abstract_df_cleaned)
print("-------------------------filter papers included based on key-based filtration ------\n")
print(df_based_pp.describe(include='all'))
print("-------------------------papers excluded based on keybased filtration---------------\n")
print(non_dl_based_pp.describe(include='all'))







######################################################################################################
# NLP-Task-1: filter deep learning based paper METHOD1: filter out all the deep learning based papers
# this includes a LLM filtering th
# requiement:
# python -m spacy download en_core_web_sm
######################################################################################################


from copy import deepcopy
import spacy
from transformers import pipeline

# Load spaCy model for tokenizing sentences
nlp = spacy.load("en_core_web_sm")

# Transformer pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification",model="valhalla/distilbart-mnli-12-3",device=0)# model="facebook/bart-large-mnli")

# Function to extract method sentences
def classify_deep_l(data, candidate_labels=["deep learning", "machine learning", "not deep learning"]):
    data_copy = data.copy(deep=True)
    data_copy["deep-learning-used"] = "not-used"
    count=0

    for index, row in data.iterrows():
        abstract = row['Abstract']
        count+=1

        print("________________________________________:", count)


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


# extract the papers which uses deep learning papers from the dataframe 
#rejected by the keybased filtering method

df_nlp_based = classify_deep_l(non_dl_based_pp[:100])
print("-------------------------filter deep learning papers based on zero shot classification----------\n")
print(df_nlp_based.head())


# NLP task1:
#1. df_dl_pp is the set paperes are deep learning based on keybased methods
#2. df_nlp_dl_pp is the set paperes are deep learning based/not deep learning 
#based added in an additional column called deep-learning used
df_dl_pp= df_based_pp.copy(deep= True)
df_nlp_dl_pp =df_nlp_based.copy(deep= True)
df_combined_dl = pd.concat([df_dl_pp, df_nlp_dl_pp], axis=0, ignore_index=True)


print("--------------------------keybased dataframe + zero-shot-classification dataframe\n")
print(df_combined_dl["deep-learning-used"].unique())
#print(df_combined_dl.describe(include='object'))
print(df_combined_dl.describe())
print("---------------------------------------------------------------------------------\n")

# set the "deep-learning-used" column as used for the dataframe extracted based on keybased filtering
import numpy as np
# Replace all instances of 'Old_Value' with 'New_Value' in 'Column_Name'
df_combined_dl['deep-learning-used']= df_combined_dl['deep-learning-used'].replace(np.nan, 'used')







#########################################################################################################
# **NLP TASK 2: For the papers deemed relevant, classify them according to the type of method used: "text
# mining", "computer vision", "both", "other"**
#########################################################################################################



# Load spaCy model for tokenizing sentences
nlp_cv_txt_mg = spacy.load("en_core_web_sm")

# Transformer pipeline for zero-shot classification
classifier_2 = pipeline("zero-shot-classification",model="valhalla/distilbart-mnli-12-3")# model="facebook/bart-large-mnli")

# Function to extract method sentences
def classify_cv_txt_mng(filtered_data, candidate_labels=["computer vision", "image processing" , "natural language processing", "text mining","computer vision and natural language processing", "None" ]):
    data_copy = filtered_data.copy(deep=True)
    data_copy["computer_vision"] = "not-used"
    data_copy["text_mining"] = "not-used"
    data_copy["computer_vision_and_text_mining"] = "not-used"


    for index, row in data_copy.iterrows():
        abstract =  row['Abstract']
        print("=================================================================================")


        doc = nlp(abstract)
        abstract_label = []
        for sent in doc.sents:
            result = classifier(sent.text, candidate_labels)
            print(result)
            print("abstact_label: ", abstract_label)
            if  result["scores"][result["labels"].index("computer vision")] > 0.5 or result["scores"][result["labels"].index("image processing")] > 0.5 :
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
                break

            # If both conditions are met, mark the "computer_vision_and_text_mining" column
            if  0 in abstract_label and 1 in abstract_label:
                data_copy.at[index, 'computer_vision_and_text_mining'] = "used"
                abstract_label = []
                break  # Stop processing sentences once both labels are identified for efficiency


            #if result["scores"][result["labels"].index("natural language processing") ]> 0.5:
            #    data_copy.at[index, 'text_mining'] = "used"  # Assign new data to each row
                #break
    return data_copy


clean_df_combined_dl= df_combined_dl[df_combined_dl["deep-learning-used"]=="used"]

print(clean_df_combined_dl.describe())
print(df_combined_dl.describe())
classifed_pp_cv_txt_othr = classify_cv_txt_mng(clean_df_combined_dl[3000:3010])
print(classifed_pp_cv_txt_othr.head())


print("-------------------------------text mining-------------------------------------------\n")
print("text mining :", classifed_pp_cv_txt_othr.text_mining.unique())
print("-------------------------------computer vision---------------------------------------\n")
print("computer vision: ",classifed_pp_cv_txt_othr.computer_vision.unique())
print("------------------------------- computer vision and text mining-----------------------\n")
print("cv and txt mining:",classifed_pp_cv_txt_othr.computer_vision_and_text_mining.unique())







#################################################################################################
#NLP task 3 :
##Extract and report the name of the method used for each relevant paper.
##this method has used zero shot classification method
#################################################################################################


# Function to extract method sentences
def find_method(data, candidate_labels=["method", "result" ,"conclusion", "context"]):
    data_copy = data.copy(deep=True)
    data_copy["method_used"] = "none"

    method =[]
    for index, row in data.iterrows():
        abstract = row['Abstract']
        method =[]

        print("#####################################################################")


        doc = nlp(abstract)
        for sent in doc.sents:
            result = classifier(sent.text, candidate_labels)
            print(result)
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


# Extract method-related sentences
extracted_method_df = find_method(classifed_pp_cv_txt_othr)
print("----------------------------------- METHOD extracted from the combined dataframe---------------\n")
print(extracted_method_df.head())

print("-----------------------------------saving the final dataframe -----------------------------------")
extracted_method_df.to_csv('final_csv_report.csv', index=False) 


























