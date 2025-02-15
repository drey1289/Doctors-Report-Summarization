import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import(
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer
)
from torch import optim

class clinical_note(Dataset):
    def __init__(self,
        data,
        tokenizer,
        column_name,
        text_max_token_len = 512,
        summary_max_token_len = 128):
     
        """ 
          ** input text can be Assesment or Assesment_plus_Subjective_Section
       
          Constructor for  Clinical_note class  
      
            Parameters:
            data - The dataset used
            tokenizer - The respective tokenizer used for encoding of data
            column_name - Name of the column to be used from dataset for encoding
            text_max_token_len - Maximum number of token to be used from input text.
            summary_max_token_len -  Maximum number of tokens to be used from summary data
            
        """
        # initializing the variables   
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
        self.column_name = column_name

    # This will return the length of dataset.
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
    __getitem__ will return the items on the given index
    
    parameters:
        index : The index of the item to be fetched.
        
    Return:
        It will be returning a dictionary containing below keys and values:
        text - The input text dataset
        summary - The summary part of dataset
        text_input_ids - Tokenized input text data
        text_attention_mask - Attension mask of tokenized  input text data
        labels - Tokenized summary data
        labels_attention_mask - Attention mask of tokenized summary data
        
        """
    
        # Encoding the input text data using the tokenizer
        data_row = self.data.iloc[index]
    
        text = data_row[self.column_name]
        text_encoding = self.tokenizer(
          text,
          max_length=self.text_max_token_len,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          add_special_tokens=True,
          return_tensors="pt"
        )
    
    
        # Encoding the summary data using the tokenizer
        summary_encoding = self.tokenizer(
          data_row["summary"],
          max_length=self.summary_max_token_len,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          add_special_tokens=True,
          return_tensors="pt"
        )

        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100

        # Returns the dictionary which contains input and output
        return dict(
          text=text,
          summary=data_row["summary"],
          text_input_ids=text_encoding["input_ids"].flatten(),
          text_attention_mask=text_encoding["attention_mask"].flatten(),
          labels=labels.flatten(),
          labels_attention_mask=summary_encoding["attention_mask"].flatten()
        )


# This class inherits from 'pl.LightningDataModule', it is a PyTorch Lightening class which gives us a standard interface for loading the data to the PyTorch Model.
class clinical_note_module(pl.LightningDataModule):
    
    """
    Constructor for the clinical_note_module class
    
    Parameters:
        train_df - Training dataset
        val_df - Validation dataset
        tokenizer - Tokenizer (Either T5 or BART Tokenizer)
        column_name - Respective Column name ( Either Assesment or Assesment_Plus_Subjective_Section)
        batch_size - batch size configuration
        text_max_token_len - maximum token length of input text data
        summary_max_token_len - Maximum length of summary data
        
    Returns:
        two dataloaders for training and validation which contains the following:
        - corresponding dataset(Either training or validation dataset)
        - batch size
        - shuffling (Whether enabled or not)
        - number of workers
            
    """
    def __init__(self,
        train_df,
        val_df,
        tokenizer,
        column_name,
        batch_size = 8,
        text_max_token_len = 512,
        summary_max_token_len = 128):
        super().__init__()
    
     # Initializing the variable
        self.train_df = train_df
        self.val_df  = val_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
        self.column_name = column_name

# Methods used for setting up of datasets for training and validation
    def setup(self, stage = None):
    
        # Creating the training dataset by calling clinical_note class
        self.train_dataset = clinical_note(
          self.train_df,
          self.tokenizer,                                     
          self.column_name,
          self.text_max_token_len,                            
          self.summary_max_token_len                           
        )
    
        # Creating the validation dataset by calling clinical_note class
        self.val_dataset = clinical_note(
          self.val_df,                                        
          self.tokenizer,                                     
          self.column_name,
          self.text_max_token_len,                            
          self.summary_max_token_len                          
        )
        
    #Method to creating the dataloader for training dataset 
    def train_dataloader(self):
        return DataLoader(
      self.train_dataset,                                 
      batch_size=self.batch_size,                         
      shuffle = True,       # Enabling shuffling of data at the begning of each epoch                                
      num_workers=2          # Number of workers required for loading the data                             
    )

    #Method to creating the dataloader for validation dataset 
    def val_dataloader(self):
        return DataLoader(
      self.val_dataset,                                    
      batch_size=self.batch_size,                          
      shuffle = False,       # Disabled shuffling of data at the begning of each epoch                                                              
      num_workers=2     # Number of workers required for loading the data               
    )


#clinical_note_model inherits from pl.LightningModule 

class clinical_note_model(pl.LightningModule):
    """
    Constructor for the class clinical_note_model
    
    Parameters:
        model_config -  which can be either BartForConditionalGeneration or T5ForConditionalGeneration
        
        model name - corresponding model (T5/BART)
        
        optimizer_name - Initialized to Adam
        
    """
    
    def __init__(self,model_config, model_name):
        super().__init__()
        self.optimizer_name = 'Adam'
        self.model = model_config.from_pretrained(model_name, return_dict = True)

# It is the forward pass, where the actual computation of model will happen for the given input
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels = None):
        output = self.model(
            input_ids,
            attention_mask = attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return output.loss, output.logits

    # In this function a single training step will be performed on a batch of data
    def training_step(self, batch, batch_ids):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
    
        loss, outputs = self(input_ids = input_ids, attention_mask=attention_mask,
             decoder_attention_mask = labels_attention_mask,labels = labels)
        
        self.log("train_loss", loss, prog_bar=True, logger=True) # Here the loss will be logged to Lightening logger
        return loss

    # In this function a single validation step will be performed on a batch of data
    def validation_step(self, batch, batch_ids):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
    
        loss, outputs = self(input_ids = input_ids, attention_mask=attention_mask,
             decoder_attention_mask = labels_attention_mask,labels = labels)
        
        self.log("val_loss", loss, prog_bar=True, logger=True) #Here the loss will be logged to Lightening logger
        return loss

    # configure_optimizers function sets the optimizer for the model
    def configure_optimizers(self)-> optim.Optimizer:
        optimizer = getattr(optim, self.optimizer_name)(self.model.parameters(), lr = 0.0001)
        return optimizer