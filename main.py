from copy import deepcopy
import spacy
from transformers import pipeline
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:,.0f}'.format
import time

from utils_for_title_extraction import *


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
df_pmid_title_abstract = df#[['PMID', 'Title', 'Abstract']]
df_pmid_title_abstract.isna().sum()
#print("length of rows: ", df_pmid_title_abstract.shape[0])
#print(df_pmid_title_abstract.describe(include='all'))


#make a new dataframe after removing rows with NaN in the 'Abstract' column
pmid_title_abstract_df_cleaned = df_pmid_title_abstract.dropna(subset=['Abstract'])

# Display the first few rows of the cleaned DataFrame
pmid_title_abstract_df_cleaned.head()
#print(pmid_title_abstract_df_cleaned.isna().sum())
#print("length of rows: ", pmid_title_abstract_df_cleaned.shape[0])



#  Make a new dataframe withh PMID, Title, Abstract , where NaN is present, which can be used later on for
#  for filtration process seperately  
df_abstract_only_nan = df_pmid_title_abstract[df_pmid_title_abstract['Abstract'].isna()]
print("length of row", df_abstract_only_nan.shape[0])
print(df_abstract_only_nan.isna().sum())
#print(df_abstract_only_nan.head())




######################################################################################################
# NLP-Task-1: filter deep learning based paper METHOD1: filter out all the deep learning based papers
######################################################################################################

def contains_keyword(row, keywords_lower):
    """
    This helper function checks if any of the specified keywords are present in the 'Abstract' or 'Title' of a research paper entry.
    It is designed for use within the `keybased_search` function to help classify papers as deep learning (DL)-based or non-DL-based.

    Parameters:
        row (Series): A pandas Series representing a single row of the DataFrame, containing at least the 'Abstract' and 'Title' columns.
        keywords_lower (list of str): A list of keywords, all in lowercase, to search for in the 'Abstract' or 'Title'.

    Returns:
        bool: Returns True if any of the keywords are found in the 'Abstract' or, if the 'Abstract' is missing (NaN), in the 'Title'.
              Returns False if none of the keywords are found.
    """


    # Use abstract if available, otherwise use title
    text_to_search = str(row['Abstract']) if pd.notna(row['Abstract']) else str(row['Title'])
    text_lower = text_to_search.lower()  # Convert to lowercase for case-insensitive matching
    return any(keyword in text_lower for keyword in keywords_lower)
  

def keybased_search(data):

    '''
    This function filters a dataset of research papers into two categories: deep learning (DL)-based papers and non-DL-based papers. 
    It searches for specific keywords within each paper's abstract or title to determine if the paper is DL-related.

    Parameters:
        data (DataFrame): A pandas DataFrame containing research paper information with columns 'Abstract' and 'Title' , 'PMID'.

    Returns:
        dl_based_papers (DataFrame): A DataFrame containing papers identified as DL-based by the presence of certain keywords.
        non_dl_based_papers (DataFrame): A DataFrame containing papers not identified as DL-based.
    '''



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




# Load spaCy model for tokenizing sentences
nlp = spacy.load("en_core_web_sm")

# Transformer pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification",model="valhalla/distilbart-mnli-12-3",device=0)# model="facebook/bart-large-mnli")

# Function to extract method sentences
def classify_deep_l(data, candidate_labels=["deep learning", "machine learning", "not deep learning"]):


    """
    This function classifies research papers as "deep learning" or "machine learning" and "not deep learning" based on the content 
    of their abstracts.It uses a natural language processing (NLP) model to assess whether each sentence in an abstract matches the 
    candidate labels with a confidence threshold, updating the classification of each paper accordingly.

    Parameters:
        data (DataFrame): A pandas DataFrame containing research paper abstracts in the 'Abstract' column.
        
        candidate_labels (list of str): A list of labels used for classification, with default values 
                                        ["deep learning", "machine learning", "not deep learning"].

    Returns:
        data_copy (DataFrame): A modified copy of the original DataFrame with an additional column 'deep-learning-used', 
                               indicating whether the abstract contains deep learning or machine learning content 
                               ("used" if it does, "not-used" otherwise).
    """


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
            #print(result)
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

df_nlp_based = classify_deep_l(non_dl_based_pp[:])
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
#print(df_combined_dl.describe())
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
#nlp_cv_txt_mg = spacy.load("en_core_web_sm")

# Transformer pipeline for zero-shot classification
#classifier_2 = pipeline("zero-shot-classification",model="valhalla/distilbart-mnli-12-3")# model="facebook/bart-large-mnli")

# Function to extract method sentences

def classify_cv_txt_mng(filtered_data, candidate_labels=["computer vision", "image processing" , "natural language processing", "text mining","computer vision and natural language processing", "others" ]):
    

    """
    
    This function classifies research papers based on their abstract content into categories related to computer vision, text mining, both, or others.
    It uses a natural language processing (NLP) model to assess whether each sentence in an abstract matches specific candidate labels.

    Parameters:
        filtered_data (DataFrame): A pandas DataFrame containing research paper abstracts in the 'Abstract' column.
        candidate_labels (list of str): A list of classification labels, defaulting to categories relevant to computer vision, text mining, 
                                        computer vision_and_text mining, image_processing and an "others" category.

    Returns:
        data_copy (DataFrame): A modified copy of the original DataFrame with four additional columns:
            - 'computer_vision': Marks "used" if the abstract contains computer vision or image processing-related content.
            - 'text_mining': Marks "used" if the abstract contains text mining or NLP-related content.
            - 'computer_vision_and_text_mining': Marks "used" if both computer vision and text mining content are present in the abstract.
            - 'others': Marks "used" if none of the other categories are identified in the abstract.
    """





    data_copy = filtered_data.copy(deep=True)
    data_copy["computer_vision"] = "not-used"
    data_copy["text_mining"] = "not-used"
    data_copy["computer_vision_and_text_mining"] = "not-used"
    data_copy["others"] = "not-used"


    for index, row in data_copy.iterrows():
        abstract =  row['Abstract']
        print("===========================INDEX========================================",index )


        doc = nlp(abstract)
        abstract_label = []
        for sent in doc.sents:
            result = classifier(sent.text, candidate_labels)
            #print(result)
            print("abstact_label: ", abstract_label)
            
            if  0 in abstract_label and 1 in abstract_label:
                data_copy.at[index, 'computer_vision_and_text_mining'] = "used"

                #print("++++++++++++++++++++both+++++++++++++++++++++++", abstract_label)
                time.sleep(3)
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


clean_df_combined_dl= df_combined_dl[df_combined_dl["deep-learning-used"]=="used"]

print(clean_df_combined_dl.describe())
print(df_combined_dl.describe())
classifed_pp_cv_txt_othr = classify_cv_txt_mng(clean_df_combined_dl[:])
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

    """
    This function extracts sentences classified as describing the "method" from the abstract of each research paper.
    It uses a natural language processing (NLP) model to classify each sentence in an abstract and collects sentences 
    that match the "method" label.

    Parameters:
        data (DataFrame): A pandas DataFrame containing research paper abstracts in the 'Abstract' column.
        candidate_labels (list of str): A list of classification labels, defaulting to ["method", "result", "conclusion", 
        "context"].

    Returns:
        data_copy (DataFrame): A modified copy of the input DataFrame with an additional 'method_used' column containing 
                               sentences from each abstract classified as "method" content.
    """




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
        #print("method used :" , joined_method)
        data_copy.at[index, 'method_used'] = joined_method


    return data_copy


# Extract method-related sentences
extracted_method_df = find_method(classifed_pp_cv_txt_othr)
print("----------------------------------- METHOD extracted from the combined dataframe---------------\n")
print(extracted_method_df.head())

print("-----------------------------------saving the final dataframe -----------------------------------")
extracted_method_df.to_csv('./results/final_csv/final_csv_report.csv', index=False) 



#========================================================================================================#
#======================================VISUALIZATION=====================================================#
print("======================================VISUALIZATION=============================================\n")
count_used_deep_learning = (extracted_method_df['deep-learning-used']=="used").sum()

print(count_used_deep_learning)
count_used_computer_vision = extracted_method_df['computer_vision'].sum()

print(count_used_computer_vision)



# Load spaCy model for tokenizing sentences
nlp = spacy.load("en_core_web_sm")

# Transformer pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification",model="valhalla/distilbart-mnli-12-3")# model="facebook/bart-large-mnli")

#===================================filtration based on Titles====================================================================#
df_only_title_df = df_abstract_only_nan.copy(deep=True)
title_df_based_pp, title_non_dl_based_pp = title_keybased_search(df_only_title_df)
print("-------------------------filter papers included based on key-based filtration ------\n")
print(title_df_based_pp.describe(include='all'))
print("-------------------------papers excluded based on keybased filtration---------------\n")
print(title_df_based_pp.describe(include='all'))

#=======================================================================================================#
title_df_nlp_based = title_classify_deep_l(title_non_dl_based_pp[:])
print("-------------------------filter deep learning papers based on zero shot classification----------\n")
print(title_df_nlp_based.head())

title_df_dl_pp= title_df_based_pp.copy(deep= True)
title_df_nlp_dl_pp =title_df_nlp_based.copy(deep= True)

title_df_combined_dl = pd.concat([title_df_dl_pp, title_df_nlp_dl_pp], axis=0, ignore_index=True)
# Replace all instances of 'Old_Value' with 'New_Value' in 'Column_Name'
title_df_combined_dl['deep-learning-used']= title_df_combined_dl['deep-learning-used'].replace(np.nan, 'used')



title_clean_df_combined_dl= title_df_combined_dl[title_df_combined_dl["deep-learning-used"]=="used"]

title_classifed_pp_cv_txt_othr = title_classify_cv_txt_mng(title_clean_df_combined_dl[:])
print(title_classifed_pp_cv_txt_othr.head())


# Extract method-related sentences from titles
title_extracted_method_df = title_find_method(title_classifed_pp_cv_txt_othr)
print("----------------------------------- METHOD extracted from the combined dataframe---------------\n")
print(title_extracted_method_df.head())

print("-----------------------------------saving the final dataframe -----------------------------------")
title_extracted_method_df.to_csv('title_final_csv_report.csv', index=False) 


title_abstract_full_combined_df =pd.concat([extracted_method_df, title_extracted_method_df], 
                                           axis=0, 
                                           ignore_index=True) 


print("-----------------------------------saving the final dataframe -----------------------------------")
title_abstract_full_combined_df.to_csv('./results/final_csv/full_final_csv_report.csv', index=False) 


print("---------------------------------title-abstract--------------------------------------------------------------\n")
full_count_used_deep_learning = (title_abstract_full_combined_df['deep-learning-used']=="used").sum()
print(full_count_used_deep_learning)

full_count_used_computer_vision = (title_abstract_full_combined_df['computer_vision']=="used").sum()
print(full_count_used_computer_vision)

full_count_used_text_mining = (title_abstract_full_combined_df['text_mining']=="used").sum()
print(full_count_used_text_mining)

full_both = (title_abstract_full_combined_df['computer_vision_and_text_mining']=="used").sum()
print(full_both)

full_others = (title_abstract_full_combined_df['others']=="used").sum()
print(full_others)
print("------------------------------------------------------------------------------------------------\n")



import matplotlib.pyplot as plt


"""
# Assuming 'df' is your DataFrame and 'column_name' is the name of the categorical column
ax = extracted_method_df['deep-learning-used'].value_counts().plot(kind='bar')
# Customize the plot
plt.title('Distribution of deep-learning-used')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate category labels for better readability if needed
# Show the plot
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')


plt.savefig("./results/plots/total_number_of_deep_learing_papers.png") 
plt.show()


count_used_computer_vision = (extracted_method_df['computer_vision']=="used").sum()
print(count_used_computer_vision)
# Assuming 'df' is your DataFrame and 'column_name' is the name of the categorical column
ax = extracted_method_df['computer_vision'].value_counts().plot(kind='bar')
# Customize the plot
plt.title('Distribution of computer_vision')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate category labels for better readability if needed
# Show the plot
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')

plt.savefig("./results/plots/total_number_of_deep_learning_papers_using_computer_vision.png") 

plt.show()



count_used_text_mining = (extracted_method_df['text_mining']=="used").sum()
print(count_used_text_mining)
# Assuming 'df' is your DataFrame and 'column_name' is the name of the categorical column
ax = extracted_method_df['text_mining'].value_counts().plot(kind='bar')
# Customize the plot
plt.title('Distribution of text mining')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate category labels for better readability if needed
# Show the plot
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')
plt.savefig("./results/plots/total_number_of_deep_learning_papers_using_text_mining.png") 

plt.show()



both = (extracted_method_df['computer_vision_and_text_mining']=="used").sum()
print(both)
# Assuming 'df' is your DataFrame and 'column_name' is the name of the categorical column
ax = extracted_method_df['computer_vision_and_text_mining'].value_counts().plot(kind='bar')
# Customize the plot
plt.title('Distribution of computer_vision_and_text_mining')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate category labels for better readability if needed
# Show the plot
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')
plt.savefig("./results/plots/total_number_of_deep_learning_papers_using_computer_vision_and_text_mining.png") 
plt.show()


others = (extracted_method_df['others']=="used").sum()
print(others)
# Assuming 'df' is your DataFrame and 'column_name' is the name of the categorical column
ax = extracted_method_df['others'].value_counts().plot(kind='bar')
# Customize the plot
plt.title('Distribution of Others')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate category labels for better readability if needed
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')
plt.savefig("./results/plots/total_number_of_deep_learning_papers_using_other_methods.png") 

# Show the plot
plt.show()



"""


#====================================================================================================#


# Assuming 'df' is your DataFrame and 'column_name' is the name of the categorical column
ax = title_abstract_full_combined_df['deep-learning-used'].value_counts().plot(kind='bar')
# Customize the plot
plt.title('full Distribution of deep learning')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate category labels for better readability if needed
# Show the plot
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')
plt.savefig("./results/plots/full_dataset_total_number_of_deep_learing_papers.png") 
plt.show()


# Assuming 'df' is your DataFrame and 'column_name' is the name of the categorical column
ax = title_abstract_full_combined_df['computer_vision'].value_counts().plot(kind='bar')
# Customize the plot
plt.title('full Distribution of computer_vision')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate category labels for better readability if needed
# Show the plot
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')
plt.savefig("./results/plots/full_total_number_of_deep_learning_papers_using_computer_vision.png") 

plt.show()




# Assuming 'df' is your DataFrame and 'column_name' is the name of the categorical column
ax = title_abstract_full_combined_df['text_mining'].value_counts().plot(kind='bar')
# Customize the plot
plt.title('full Distribution of text mining')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate category labels for better readability if needed
# Show the plot
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')
plt.savefig("./results/plots/full_total_number_of_deep_learning_papers_using_text_mining.png") 

plt.show()




# Assuming 'df' is your DataFrame and 'column_name' is the name of the categorical column
ax = title_abstract_full_combined_df['computer_vision_and_text_mining'].value_counts().plot(kind='bar')
# Customize the plot
plt.title('full  Distribution of computer_vision_and_text_mining')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate category labels for better readability if needed
# Show the plot
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')
plt.savefig("./results/plots/full_total_number_of_deep_learning_papers_using_computer_vision_and_text_mining.png") 

plt.show()



# Assuming 'df' is your DataFrame and 'column_name' is the name of the categorical column
ax = title_abstract_full_combined_df['others'].value_counts().plot(kind='bar')
# Customize the plot
plt.title('full Distribution of Others')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate category labels for better readability if needed

# Show the plot
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')

plt.savefig("./results/plots/full_total_number_of_deep_learning_papers_using_other_methods.png") 

plt.show()


#----------------------------------------------------------------------------------------------------------------------------#

# DATASET FULL STATISTICS VIUSALIZATION

# Specify the columns to analyze
selected_columns = ['deep-learning-used', 'computer_vision', 'text_mining', 'computer_vision_and_text_mining', 'others']

# Count the occurrences of "used" in each selected column
category_counts = title_abstract_full_combined_df[selected_columns].apply(lambda x: x.value_counts().get('used', 0))

# Plotting
plt.figure(figsize=(10, 6))
ax = category_counts.plot(kind='bar')

# Customize the plot
plt.title("Frequency of used 'deep-learning-used', 'computer_vision', 'text_mining', 'computer_vision_and_text_mining', 'others")
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

# Annotate the bars with frequency values
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig("./results/plots/data_statistics.png") 

# Show the plot
plt.show()












































