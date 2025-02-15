**Doctor's-Report-Summarization**

Doctors' Report Summarization using BART &amp; T5 – A deep learning project leveraging transformers to generate concise summaries of medical reports, enhancing efficiency for healthcare professionals. Utilizing NLP to summarize patients’ significant issues from daily progress notes can optimize information overload in hospital management and assist contributors with computerized diagnostic decision support. Health records require a system model to understand, outline, and summarize problem lists.

This project utilizes BART and T5 transformer models to generate concise, high-quality summaries of doctors' reports, enabling faster and more efficient decision-making.

🔹 Audience:

The healthcare providers ( Doctors and physicians)  

🔹 Tech Stacks:

NLP Models: BART, T5
Libraries: Transformers, PyTorch, Hugging Face, Scikit-learn, Pandas
Programming Language: Python

🔹 Data format:

Dataset: MIMIC-III Clinical Notes (Link to data wasn't provided because of data privacy and policy issues as it is an healthcare data.) But the link to request for the data is: https://archive.physionet.org/physiobank/database/mimic3cdb/

Below is how the data appears in a csv format:

![kgk](https://github.com/user-attachments/assets/2f0315ad-210e-49c8-acc2-d6ea4d4f6af9)

1. File_id: Unique identifier for each record (anonymized or synthetic ID).
2. Assessment: A brief medical assessment by the physician.
3. Subjective: Patient-reported symptoms and relevant history.
4. Objective: Observations and measurements taken by the medical team (e.g., vitals, physical exam findings).
5. Summary: A concise, human-written or existing summary (optional) used for model training/evaluation.

🔹 Features:

Preprocessing of medical reports
Fine-tuning BART & T5 for summarizationEvaluation using ROUGE & BLEU scores
Comparison of both models' performance
Interactive Jupyter notebooks for experimentation

🔹 Installation Instructions:

To execute the code following steps need to be done:  

1. Upload all 5 in the code folder into your google drive including the CSV file, which is the input file. If you rename the files, need to change the name accordingly in the code.  
2. Upload the input file with the name BioNLP2023-1A-Train.csv, with column names ‘File ID’, ‘Assessment’, ‘Summary’,  ‘Subjective Sections’, and ‘Objective Sections’. 
3. A Google Colab pro account is required to run the code as it requires 38 GPU and 57 GB memory if we need to execute the whole code.  
4. Open google colab and first run the Exploratory Data Analysis.ipynb file, in which you will get an idea of the text and summary.  
5. Upload T5.ipynb and BART.ipynb files in google colab. Run the code.  
6. You will get the generated summary file in the drive and download it to get the predicted summary.  
7. The image below shows the folder structure in drive while the predicted file is saved in the drive. bart_base_A.csv in the folder belongs to the result final output file of the BART base model with Assessment as the input.

![image](https://github.com/user-attachments/assets/708e7b37-d0c1-4853-b9ed-d10c4c77e703)

🔹 Evaluation metrics:
1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)ROUGE-1, ROUGE-2, ROUGE-L: Measures n-gram overlap and sequence similarity between the generated summary and reference text.
2. BERTScore: Uses pretrained contextual embeddings (e.g., BERT) to compare similarity between generated and reference texts. Also provides Precision, Recall, and F1 scores based on token-level embedding similarity which i used.

🔹 Results:

After doing all the evaluation steps, it is found that T5-Large with assessment and subjective section is giving a better ROUGE-L score while BART-base with Assessment gives a better Bert-Score.

🔹 Future Strategy:

There is a plan to make this into a web application, at which point it intends to provide online assistance.
