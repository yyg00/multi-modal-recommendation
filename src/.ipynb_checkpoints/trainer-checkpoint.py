import glob
import logging
import os
import random
from typing import Callable, Dict, List, Union, Tuple, Any
import torch
from datasets import load_dataset
from openmatch.trainer import DRTrainer
from torch.utils.data import Dataset, IterableDataset
from transformers import BatchEncoding, PreTrainedTokenizer, DataCollatorWithPadding
from openmatch.dataset.train_dataset import TrainDatasetBase, StreamTrainDatasetMixin, MappingTrainDatasetMixin
from dataclasses import dataclass


@dataclass
class TasteCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128
    len_seq: int = 2
    max_img_len: int = 256 # maximum image embedding length

    def __call__(self, features):
        qq = [f["query_"] for f in features]
        dd = [f["passages"] for f in features]
        if isinstance(dd[0], list):
            dd = sum(dd, [])          
        q_collated_list = list()
        for seq in qq:
            q_collated = self.tokenizer.pad(
                seq,
                padding='max_length',
                max_length=self.max_q_len,
                return_tensors="pt",
            )
            q_collated_list.append(q_collated)
        # images for queries and passages
        qq_img = [f["query_img"] for f in features]
        dd_img = [f["passages_img"] for f in features]
        if isinstance(dd_img[0], list):
            dd_img = sum(dd_img, []) # concatenate negatives for each sample
        seq_input_ids = []
        seq_attention_mask = []
        for q_collated in q_collated_list:
            item_input_ids = q_collated.data['input_ids']
            item_attention_mask = q_collated.data['attention_mask']
            cur_item = item_input_ids.size(0)
            if cur_item < self.len_seq:
                b = self.len_seq - cur_item
                l = item_input_ids.size(1) #256
                pad = torch.zeros([b, l], dtype=item_input_ids.dtype)
                item_input_ids = torch.cat((item_input_ids, pad), dim=0)
                item_attention_mask = torch.cat((item_attention_mask, pad), dim=0)
            seq_input_ids.append(item_input_ids[None])
            seq_attention_mask.append(item_attention_mask[None])
        seq_input_ids = torch.cat(seq_input_ids,dim=0)
        seq_img_features = []
        for i in range(len(qq_img)): # for each query, padding/truncating image embedding..?
            q_img = torch.tensor(qq_img[i], dtype=torch.float32)
            qqq = q_img.reshape(1,-1)
            imgl = qqq.size(1)
            if imgl < self.max_img_len:
                padding = torch.zeros(1, self.max_img_len - imgl)
                truncated_tensor = torch.cat([qqq, padding], dim=1)
            else:
                truncated_tensor = qqq[:, :self.max_img_len]
            seq_img_features.append(truncated_tensor)
        seq_img_features = torch.cat(seq_img_features, dim=0)
        seq_img_features = torch.unsqueeze(seq_img_features, 1) # [batch_size, 1, max_img_len]
        
        seq_attention_mask = torch.cat(seq_attention_mask,dim=0)
        query = (seq_input_ids,seq_attention_mask, seq_img_features)
        
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        item_input_ids = d_collated.data['input_ids']
        item_attention_mask = d_collated.data['attention_mask']
        item_input_ids = torch.unsqueeze(item_input_ids, 1)
        item_attention_mask = torch.unsqueeze(item_attention_mask, 1)
        
        # passage imgs padding
        dd_img_features = []
        for i in range(len(dd_img)):
            d_img = torch.tensor(dd_img[i], dtype=torch.float32)
            ddd = d_img.reshape(1,-1)
            imgl = ddd.size(1)
            if imgl < self.max_img_len:
                padding = torch.zeros(1, self.max_img_len - imgl)
                truncated_tensor = torch.cat([ddd, padding], dim=1)
            else:
                truncated_tensor = ddd[:, :self.max_img_len]
            dd_img_features.append(truncated_tensor)
        dd_img_features = torch.cat(dd_img_features, dim=0)
        dd_img_features = torch.unsqueeze(dd_img_features, 1)
        item = (item_input_ids,item_attention_mask, dd_img_features)


        return query, item



class TasteTrainer(DRTrainer):
    def __init__(self,*args, **kwargs):
        super(TasteTrainer, self).__init__(*args, **kwargs)
    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for tup in inputs:
            x = tup[0].to(self.args.device)
            y = tup[1].to(self.args.device)
            z = tup[2].to(self.args.device) # images
            prepared.append((x,y,z))
        return prepared
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.distillation:
            if self.args.distil_mode == "pairwise":
                query, positive, negative, score = inputs
                outputs = model(query=query, positive=positive, negative=negative, score=score)
            else:  # listwise
                query, passage, score = inputs
                outputs = model(query=query, passage=passage, score=score)
        else:
            query, passage = inputs
            outputs = model(query=query, passage=passage)
        # print(outputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

class TasteTrainDataset(TrainDatasetBase):

    def create_one_example(self, text_encoding: List[int], is_query=False) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item



    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            encoded_query = []
            qry = example['query']
            for item in qry:
                encoded_query.append(self.create_one_example(item,True))
            encoded_passages = []
            encoded_passages_imgs = []
            group_positives = example['positives']
            group_negatives = example['negatives']
            images = example['query_photos'] # image sequences
            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0] 
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]
            encoded_passages.append(self.create_one_example([int(i) for i in pos_psg[0]]))
            encoded_passages_imgs.append(pos_psg[1]) # positive passage image
            negative_size = self.data_args.train_n_passages - 1
            if len(group_negatives) < negative_size:
                if hashed_seed is not None:
                    negs = random.choices(group_negatives, k=negative_size)
                else:
                    negs = [x for x in group_negatives]
                    negs = negs * 2
                    negs = negs[:negative_size]
            elif self.data_args.train_n_passages == 1:
                negs = []
            elif self.data_args.negative_passage_no_shuffle:
                negs = group_negatives[:negative_size]
            else:
                _offset = epoch * negative_size % len(group_negatives)
                negs = [x for x in group_negatives]
                if hashed_seed is not None:
                    random.Random(hashed_seed).shuffle(negs)
                negs = negs * 2
                negs = negs[_offset: _offset + negative_size]

            for neg_psg in negs:
                encoded_passages.append(self.create_one_example([int(i) for i in neg_psg[0]]))
                encoded_passages_imgs.append(neg_psg[1]) # neg passage image

            assert len(encoded_passages) == self.data_args.train_n_passages
            # pass passage and query images
            return {"query_": encoded_query, "passages": encoded_passages, "passages_img": encoded_passages_imgs, "query_img": images}  # Avoid name conflict with query in the original dataset

        return process_fn


class StreamDRTrainDataset(StreamTrainDatasetMixin, TasteTrainDataset):
    pass


class MappingDRTrainDataset(MappingTrainDatasetMixin, TasteTrainDataset):
    pass
