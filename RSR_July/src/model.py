"""
Model Architecture Module for Representational Similarity Regularisation (RSR)

This module defines the custom RoBERTa architecture wrapper required for 
RSR fine-tuning. It isolates the target span embeddings, applies mean-pooling, 
and extracts them for similarity ranking.

Workflow Context:
- Consumed by: `train.py` (to instantiate the model) and `evaluate.py` (to extract vectors during benchmark testing).
- Relies on: HuggingFace's `transformers` library for the base `roberta-base` weights.

Classes:
- RobertaRSRModel: Wraps HuggingFace's RobertaForMaskedLM, adding target vector extraction.
"""

import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM

class RobertaRSRModel(nn.Module):
    """
    A custom wrapper around `RobertaForMaskedLM` that enables extraction of 
    standardized target vectors for similarity regularisation.
    
    Workflow Context:
    This model sits at the center of the training loop in `train.py`. It is 
    called during both Masked Language Modeling (MLM) steps via `forward()`, 
    and Representational Similarity Regularisation (RSR) steps via `target_vectors()`.
    
    Attributes:
        roberta_mlm (RobertaForMaskedLM): The underlying pre-trained language model.
    """
    def __init__(self, model_name="roberta-base", rsr_layer=None):
        """
        Initializes the model architecture.
        
        Args:
            model_name (str): The identifier for the HuggingFace model.
            rsr_layer (int, optional): The specific hidden layer to extract target vectors from. 
                                       If None, extracts from the last hidden state.
        """
        super().__init__()
        # Load the base model with its standard Masked Language Modeling (MLM) head
        self.roberta_mlm = RobertaForMaskedLM.from_pretrained(model_name)
        self.rsr_layer = rsr_layer
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Standard forward pass for Masked Language Modeling (MLM).
        
        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention mask.
            labels (Tensor): Target labels for masked tokens.
            
        Returns:
            outputs (MaskedLMOutput): HuggingFace output object containing the MLM loss.
        """
        return self.roberta_mlm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
    def target_vectors(self, input_ids, attention_mask, target_spans, apply_standardisation=False):
        """
        Extracts, mean-pools, and optionally standardizes target word vectors from a batch.
        
        This method performs the core RSR forward pass:
        1. Passes the sequence through the RoBERTa encoder.
        2. Slices out the hidden states corresponding to the subwords of the target word.
        3. Mean-pools those subword states into a single vector (768-d).
        4. Applies batch-wise Z-score standardization directly on the pooled representations (if True).
        
        Args:
            input_ids (Tensor): Tensor of shape (batch_size, seq_len).
            attention_mask (Tensor): Tensor of shape (batch_size, seq_len).
            target_spans (list of tuple): List of tuples [(start_idx, end_idx), ...] 
                                          indicating where the target word lives in each sequence.
            apply_standardisation (bool): Whether to apply batch-wise Z-score standardisation.
                                          Defaults to False (disabled during regular training/eval).
            
        Returns:
            Tensor: The vectors of shape (batch_size, hidden_size).
        """
        # Forward pass through the base encoder to get contextual hidden states
        if self.rsr_layer is not None:
            outputs = self.roberta_mlm.roberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.rsr_layer]
        else:
            outputs = self.roberta_mlm.roberta(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state # Shape: (batch_size, seq_len, hidden_size)
        
        batch_size = hidden_states.size(0)
        pooled_vectors = []
        
        # Iterate over the batch to extract and pool the specific target spans
        for i in range(batch_size):
            start_idx, end_idx = target_spans[i]
            
            # Slice out the specific subword tokens for the target word
            span_hidden = hidden_states[i, start_idx:end_idx, :]
            
            # Mean-pool the subword vectors into a single vector
            mean_pooled = torch.mean(span_hidden, dim=0)
            pooled_vectors.append(mean_pooled)
            
        pooled_vectors = torch.stack(pooled_vectors) # Shape: (batch_size, hidden_size)
        
        # Apply Z-score normalization across the batch for each dimension.
        # This prevents dimensions from exploding or collapsing during similarity training.
        if apply_standardisation and batch_size > 1:
            mean = pooled_vectors.mean(dim=0, keepdim=True)
            std = pooled_vectors.std(dim=0, keepdim=True) + 1e-8
            standardised = (pooled_vectors - mean) / std
            return standardised
        return pooled_vectors
