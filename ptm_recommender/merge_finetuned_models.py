import os
from copy import deepcopy


class merge_models:

    def __init__(self, ptm, ptm_config, ptm_tokenizer, ftm_1, ftm_2, ratio, out_dir):
        self.ptm = ptm
        self.ptm_config = ptm_config
        self.ptm_tokenizer = ptm_tokenizer
        self.ftm_1 = ftm_1
        self.ftm_2 = ftm_2
        self.ratio = ratio
        self.out_dir = out_dir

        self.merge()

    # get weight names
    def get_weight_nm(self):
        weight_names = []
        for k, v in self.ptm.named_parameters():
            if k.endswith('.weight'):
                # if 'encoder' in k or 'transformer' in k:  # "transformer" is for "distilbert-base-uncased"
                weight_names.append(k)
        return weight_names

    # save merged model
    def save(self, model):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        model.save_pretrained(self.out_dir)
        self.ptm_tokenizer.save_pretrained(self.out_dir)
        self.ptm_config.save_pretrained(self.out_dir)

    def merge(self):
        # model to be returned
        merged_model = deepcopy(self.ptm)
        # merge weights
        weight_names = self.get_weight_nm()
        for nm in weight_names:
            merged_weight = (1-self.ratio) * self.ftm_1.state_dict()[nm] + self.ratio * self.ftm_2.state_dict()[nm]
            merged_model.state_dict()[nm].data.copy_(merged_weight)

        self.save(merged_model)
