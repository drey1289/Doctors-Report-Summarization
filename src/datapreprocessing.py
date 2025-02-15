import re
import pandas as pd
from google.colab import drive
from nltk.tokenize import word_tokenize
import pytorch_lightning as pl
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import copy

class ProcessedData:
  def __init__(self):
    nltk.download('punkt')
    nltk.download("stopwords")
    nltk_stop_words = set(stopwords.words('english'))
    additional_stop_words = ['some', 'other', 'stop', 'words', 'to', 'remove', 
                            'unchnaged', 'admission', 'systems', 'PM', 'AM', 'a.m', 'p.m', 'noted', 
                            'Review', 'unchanged', '%', '>', 'Known', 'except', '2', 
                            'At', '-', 'presents', 'START', 'admitted', 'patient', 'overnight', 'old', 'year',
                            'transferred','given', 'Drug', 'Unknown', 'Changes', 'No', '1', '', 'dose', 'Plan', 'NOT','year', 'plan', '_', 'Again', 'year-old','Mr','Mr.', 'Mrs.']
    self.stop_words = nltk_stop_words.union(additional_stop_words)
    self.load_file_into_df()
    self.df['Assessment_Plus_Subjective'] = self.df['Assessment'] + ' ' + self.df['Subjective Sections']
    self.unprocessed_df = copy.deepcopy(self.df)
    self.preprocess()
 
  #Load file from drive
  def load_file_into_df(self):
    drive.mount('/content/drive')
    self.df = pd.read_csv("/content/drive/MyDrive/BioNLP2023-1A-Train.csv")

   #Removing unwanted tokens
  def scrub_unrequired_text(self, column):
    date = [r'\*\*\d{4}-\d{1,2}-\d{1,2}\*\*']
    pii_patterns = [r'\[\*\*.+?\*\*\]', r'\[.*?\]', r'\d+', r'[\'"``]', r':', r'\?', r'w/', r'=',r'yo\b', r'\([^)]*\)', r'\*\*Age over \d+\*\*\]', r'(\d+) yo', r'(\d+) y/o', r'(\d+)yo', r'(\d+)y/o', r'\b\d+[MF]\b', r"'s\b", r'\b\d+[MF]\b']
    for pattern in (date + pii_patterns):
      self.df[column] = self.df[column].apply(lambda x: re.sub(pattern, '', x))
  
    #Replace multiple spaces with a single space
    self.df[column] = self.df[column].apply(lambda x: re.sub(r'\s+', ' ', x))
    self.df[column] = self.df[column].apply(lambda x: re.sub(r'\.', ';', x))

   # Replacing patterns describing male and female
  def process_gender_text(self, column):
    male_patterns = [r'(\d+) yo (M|m)', r'(\d+) y/o (M|m)', r'(\d+)yo (M|m)', r'(\d+)y/o (M|m)', r'(\d+)M', r'(\d+) M']
    female_patterns = [r'(\d+) yo (F|f)', r'(\d+) y/o (F|f)', r'(\d+)yo (F|f)', r'(\d+)y/o (F|f)', r'(\d+)F', r'(\d+) F']
  
    for pattern in male_patterns:
      self.df[column] = self.df[column].apply(lambda x: re.sub(pattern, 'male', x))
    for pattern in female_patterns:
      self.df[column] = self.df[column].apply(lambda x: re.sub(pattern, 'female', x))

    #Removing stop word and additional stop words
  def scrub_stopwords(self, column):
    self.df[column] = self.df[column].apply(lambda x: word_tokenize(x))
    self.df[column] = self.df[column].apply(lambda x: [word for word in x if not word in self.stop_words])
    self.df[column] = self.df[column].apply(lambda x: ' '.join(x))

    #Removing consecutive duplicates
  def remove_consecutive_duplicates(self, text):
      words = text.split()
      filtered_words = [words[0]] + [word for i, word in enumerate(words[1:]) if word != words[i]]
      return ' '.join(filtered_words)

    # Dropping unwanted columns
  def drop_unrequired_columns(self):
    self.df.drop(columns = ['Subjective Sections', 'Objective Sections'], inplace = True)
    self.df.dropna()

    # Calling main preprocessing function
  def preprocess(self):
    for column in ['Assessment', 'Assessment_Plus_Subjective']:
      self.process_gender_text(column)
      self.scrub_unrequired_text(column)
      self.scrub_stopwords(column)
      self.df[column].apply(self.remove_consecutive_duplicates)
    
    self.drop_unrequired_columns()
    self.df.columns = ["File ID", "assessment", "summary", "Assessment_Plus_Subjective"]