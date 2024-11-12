

# Environment:
- install anaconda
- install GIT
- create a conda environment with python version >= 3.8
- Then install all the libararies mentioned in the Requirement



# Hardware used:
- Nividia RTX A200 : 4GB VRAM  
- It took me 6-7 hours to complete the inference and get the data statistics.

# Large language model used:
- DistilBart-MNLI("https://huggingface.co/valhalla/distilbart-mnli-12-3")

# Requiments

- ! pip install spacy
- ! pip install transformers
- ! pip install pandas
- ! pip install numpy
- ! pip install matplotlib
- Install pytorch version , with cuda 11.8 (pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)
- python -m spacy download en_core_web_sm

- Please also see the requirement.txt 
    - run in the terminal:  pip install -r requirements.txt

# Instruction to run the code

- clone the repo : git clone https://github.com/abhishekdiphu/filter_pipeline.git

- go to the folder in the terminal by: cd filter_pipeline

- create a anaconda environment : conda create -n scinext python=3.9

- activate the anaconda environment: conda activate scinext

- install all the necessary librabies: pip install -r requirements.txt

- to run the whole pipeline (Task1 + Task2 + Task3 ) , please execute the script main.py using:  "python main.py"




# Dataset:

### The dataset can be accessed at Virology AI Papers Repository (https://github.com/jd-coderepos/virology-ai-papers/). It includes a header row and multiple data rows generated from keyword-based searches. The list of keywords used for the searches is 
available here(https://docs.google.com/document/d/1uMkXik3B3rNnKLbZc5AyqWruTGUKdpJcZFZZ4euM0Aw/edit?tab=t.0#heading=h.gjdgxs).


# Tasks: Screening Task, Semantic NLP Filtering for Deep Learning Papers in Virology/Epidemiology


## Task 1:Implement semantic natural language processing techniques to filter out papers that do not meet the criteria of utilizing deep learning approaches in virology/epidemiology.

- step 1: preprocessing the dataset, and removed rows that has NaN in the Abstract column
- step 2: prepared a new dataframe , which has only Abstract  column where NAN rows are droped. 
- step 3: used keyword based filtering method (rule based) based on the keywords presnt  in *list-of-queries-to-pubmed.docx*
- step 4: using a LLM , used zeroshot classification to either deep learning, machine leanring or non deep learning
- Step 5: Prepared a new dataframe, where, we have  no abstracts.(in this case we use the title to filter out the papers)
- step 6: we repeat the steps 3- 4. 


## Task 2: For the papers deemed relevant, classify them according to the type of method used: ["text mining", "computer vision", "both", "other"].

- step 1: using a LLM , used zeroshot classification to "text mining", "computer vision", "image processsing", "text mining_and_computer vision", "others", "natural language processing"

- "computer vision", "image processsing"  is categorized as "computer vision".

- "text mining", "natural language processing"  is categorized as "text mining".

- if both computer vision and text mining exist, then , it is set to "computer_vision_and_text_mining"

- if non of them exist set it to "others" 

- Step 2 : Use the prepared dataframe where, we have  no abstracts.(in this case we use the title to filter out the papers)
- step 3 : we repeat the steps 1. 

## Task 3: Extract and report the name of the method used for each relevant paper.

- step 1: using a LLM , used zeroshot classification to "method", "result" ,"conclusion", "context"

- join sequences of sentences classified as of method class in each abstract are joined and saved in a seperate column "method_used"

- Step 2 : Use the prepared dataframe where, we have  no abstracts.(in this case we use the title to filter out the papers)
- Step 3 : we repeat the steps 1. 



# Results 
### The final rsults are stored in the folder *./results/plots* and *results/final_csv. This csv contains all the papers that has used deep leaning. Futher I have added addtional column "computer_vision", "text_mining" , "computer_vision_and_text_mining" and "others" , "deep_learning_used" , each of these addtional column has two attributes "used" & "not_used". Other then that, there is also a addtional column "method used",  which tells, what method has been used in the paper.  
- if it set to "used" it mmeans that the paper has used this method
- The graph below tells us us, how many paper have used "deep leaarining" , "computer vision" , "text mining", "computer_vision_text_mining" "others".

![Description of Image](./results/plots/data_statistics.png)




# Discusion:
## which NLP technique for filtering the papers have you used? 
- for the first task , where  papers had to be filtered based on using deep learning used in the papers, I have used a hybrid approch. initilally i have used a keybased filtering method, where filter the papers which  has the terms mentioned in the *list-of-queries-to-pubmed.docx* . Once i filtered out all the papers which  has these keywords, i am left with 4149 papers which according to it was not deep learning based papers. Then I used zero shot classification method , and filter out deep learning papers from this subset of 4149 papers.

- for the second task as well , i have used zeroshot classificstion , to futher categorized the deep learning based papers into "computer vsion" , "text mining" , "both ", "others"

- for the 3rd  task as well , i have used zeroshot classificstion , filter out the method used in  the deep learing based  papers, where i select the sequences  in the abstract, sentences that is closer to the label "method" . I just joined all these sequences in the abstract closer to the "method" class and reported in the "method_used" column in the final csv file generated. 


## Why do you think your approach will be more effective than keywords-based filtering? 
- When I used kezbased method only 7088 papers(based on abstract) were classifided as  deep learnirng based papers and 4011 aprox. were classified as not deep learning papers. but when I used LLM to futher filter these 4149 papers (based on abstract), the zeroshot classification using LLM were futher able to detect more deep learning based papers.

- since I  did not want to retrain a new LLM , and in this method I can set dynamic classification labels,  it was useful for me.

- The method also find a sementic similarity, how the each sequence in the abstract are closer to my labels  and provides me a likeihood, how close these seqeunces are to my set labels. Where as in keybased methods, it pretty deterministic, where , if keywords are present or not and there is no intelligent resoning.



## What are the resulting dataset statistics?

#### papers which abstracts are missing , i have used the title for filtration, incase abstract is present, i have used abstract for filtration process.

 - Deep learning based papers :10,211 
     - Papers that does use deep learning paper : 11,450(total) - 10,211 = 1,239 
     - Papers that have used Computer Vision:2176
     - Papers that have used Text Mining:1673
     - Papers that have used Computer vision and text mining: 43 (these is also included in "computer vision" and "text mining")
     - papers that have used other methods: 6406
     - The graph below tells us us, how many paper have used "deep leaarining" , "computer vision" , "text mining", "computer_vision_text_mining" "others".

     - papers in which abstracts are absent: 213 (in this case I have used the tile for the filtration process) 


![Description of Image](./results/plots/data_statistics.png)



