from rouge import Rouge
from bert_score import score
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import model_checkpoint,EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

class utililty():

  def display_tokens(self, tokenizer, column,train_df):
    text_token_counts = []                                                  # Initializing the list for text token count
    summary_token_counts = []                                               # Initializing the list for summary token count
    for row in train_df.iterrows():
      text_token_count = len(tokenizer.encode(row[1][column]))              #Calculates the number of tokens in the input text of a given row of a dataset using a tokenizer object
      text_token_counts.append(text_token_count)

      summary_token_count = len(tokenizer.encode(row[1]["summary"]))        #Calculates the number of tokens in the summary of a given row of a dataset using a tokenizer object
      summary_token_counts.append(summary_token_count)

    fig, (ax1, ax2) = plt.subplots(1,2)

    sns.histplot(text_token_counts, ax=ax1)
    ax1.set_title("full text token counts")

    sns.histplot(summary_token_counts, ax = ax2)
    ax2.set_title("summary text token counts")


  def trainer(self, n_epochs, loggger_name):
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    checkpoint_callback = model_checkpoint.ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min")
    logger = TensorBoardLogger("lightening_logs", name=loggger_name)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=n_epochs,
        gpus=1)
    return trainer

  def summarize(self, text, tokenizer, trained_model):
    # Tokenize the input text
    text_encoding = tokenizer(
        text,
        max_length = 512,
        padding="max_length",
        truncation = True,
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = "pt"
    )
  # Generate summary using trained model
    generated_ids = trained_model.model.generate(
        input_ids = text_encoding["input_ids"],
        attention_mask = text_encoding["attention_mask"],
        max_length=150,
        num_beams=2,
        repetition_penalty = 2.5,
        length_penalty = 1.0,
        early_stopping=True

    )
    # Decode the generated summary from token IDs to text
    pred = [
        tokenizer.decode(gen_id, skip_special_tokens = True, clean_up_tokenization_spaces = True)
        for gen_id in generated_ids
    ]

    return "".join(pred)


  def evalution_rougue_l_and_Bert_Score(self, submission_data, model_type):
      rouge = Rouge()    #ROUGUE Object
      predicted_summaries = submission_data['Prediction']     # Assigning values in prediction column to generated_summaries
      original_summaries = submission_data['summary']                 # Assigning values in summary column to reference_summaries
      scores = rouge.get_scores(predicted_summaries.str.lower(), original_summaries.str.lower(), avg=True)     # Calculating ROUGUE-L Score
      rougue_l_score = scores['rouge-l']['f']
      print('ROUGE-L score:', scores['rouge-l']['f'])                  # Printing ROUGUE-L Score
    
      # Bert Score calculation
      P, R, F1 = score(submission_data['Prediction'].str.lower().tolist(), submission_data['summary'].str.lower().tolist(), lang='en', model_type= model_type, verbose=True)
      precision = P.mean().item()      # precision
      recall = R.mean().item()         # recall
      f1_score = F1.mean().item()      # f1 score
      print(f"BERT-Precision: {P.mean().item():.6f}\nBERT-Recall: {R.mean().item():.6f}\nBERT-F1: {F1.mean().item():.6f}")    # Printing Bert score
      return rougue_l_score, precision, recall, f1_score