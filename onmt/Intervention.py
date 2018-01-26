import torch
from onmt.Translator import Translator
from debiaswe.debiaswe import nmt_emb_perturb
import numpy as np
import debiaswe.debiaswe.we as we

def make_intervention_model(model,src_dict,word_intervene,perturbed_embedding):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    intervene_idx = src_dict.stoi[word_intervene]
    ##Causal Intervention on word_embedding
    model.encoder.embeddings.word_lut.weight.data[intervene_idx,:] = torch.from_numpy(perturbed_embedding)


    return model

word = 'nurse'
E = nmt_emb_perturb.nmt_emb_perturb('./debiaswe/embeddings/src_embeddings.txt')
with open('./debiaswe/data/definitional_pairs.json', "r") as f:
    defs = json.load(f)
print("definitional", defs)
gender_direction_nmt = we.doPCA(defs, E).components_[0]
perturbed_gender = perturb_gender('nurse',gender_direction_nmt,E)

