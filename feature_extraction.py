from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Load pre-trained mBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

def get_bert_embeddings_for_long_text(text, tokenizer = tokenizer,  model = model, max_length=512):
    # Tokenize and encode the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=False, max_length=max_length)
    input_ids = inputs['input_ids'][0]  # Get the input IDs as a tensor
    attention_mask = inputs['attention_mask'][0]  # Get the attention mask

    # Calculate the number of chunks
    num_chunks = (len(input_ids) + max_length - 1) // max_length
    chunk_embeddings = []
    
    with torch.no_grad():
        for i in range(num_chunks):
            # Get the chunk of input IDs and attention mask
            chunk_input_ids = input_ids[i * max_length: (i + 1) * max_length].unsqueeze(0)
            chunk_attention_mask = attention_mask[i * max_length: (i + 1) * max_length].unsqueeze(0)

            # Get the output for this chunk
            outputs = model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
            
            # Get the embedding for the [CLS] token
            chunk_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            chunk_embeddings.append(chunk_embedding)
    
    # Average the embeddings for all chunks
    avg_embedding = np.mean(chunk_embeddings, axis=0)
    
    return avg_embedding
