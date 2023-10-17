import torch
from torch.utils.data import Dataset
import json
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self,data,tokenizer,args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args
        self.template = self.toke_template()
        self.max_img_len = 256 # need to move to args later
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    def toke_template(self):
        t_list = []
        template1 = "Here is the visit history list of user: "
        template2 = " recommend next item "
        t1 = self.tokenizer.encode(template1, add_special_tokens=False,
                                     truncation=False)
        t_list.append(t1)
        t2 = self.tokenizer.encode(template2, add_special_tokens=False,
                                   truncation=False)
        t_list.append(t2)
        return t_list
    def collect_fn(self, data):
        sequence_ids = []
        sequence_masks = []
        batch_target = []
        sequence_imgs = [] # images
        for example in data: # data: [example, eaxmple], example: [[seq], [tar], [img]]
            batch_target.append(example[1])
            seq_text = example[0]
            qq_img = torch.Tensor(example[2]) # images
            qq_img = qq_img.reshape(1,-1)
            seq = self.tokenizer.encode(seq_text, add_special_tokens=False,
                                        truncation=False)
            seq = list_split(seq,self.args.split_num)
            seq[0] = self.template[0] + seq[0] + self.template[1]
            s_ids = []
            s_masks = []
            for s in seq:
                outputs = self.tokenizer.encode_plus(
                    s,
                    max_length=self.args.seq_size,
                    pad_to_max_length=True,
                    return_tensors='pt',
                    truncation=True,
                )
                input_ids = outputs["input_ids"]
                attention_mask = outputs["attention_mask"]
                s_ids.append(input_ids)
                s_masks.append(attention_mask)
            s_ids = torch.cat(s_ids, dim=0)
            # print(s_ids.shape)
            s_masks = torch.cat(s_masks, dim=0)
            cur_item = s_ids.size(0)
            if cur_item < self.args.num_passage:
                b = self.args.num_passage - cur_item
                l = s_ids.size(1)
                pad = torch.zeros([b, l], dtype=s_ids.dtype)
                s_ids = torch.cat((s_ids, pad), dim=0)
                s_masks = torch.cat((s_masks, pad), dim=0)
            sequence_ids.append(s_ids[None])
            sequence_masks.append(s_masks[None])
            if qq_img.size(1) < self.max_img_len:
                padding = torch.zeros(qq_img.size(0), self.max_img_len - qq_img.size(1))
                final_img = torch.cat([qq_img, padding], dim=1)
            else:
                final_img = qq_img[:, :self.max_img_len]
            # print(final_img.shape)
            sequence_imgs.append(final_img)

        sequence_ids = torch.cat(sequence_ids, dim=0)#[512, num_pass, max_len / num_pass]
        sequence_masks = torch.cat(sequence_masks, dim=0)
        sequence_imgs = torch.cat(sequence_imgs, dim=0) # [512, 1, max_len]
        sequence_imgs = torch.unsqueeze(sequence_imgs, 1)

        # print(sequence_ids.shape, sequence_imgs.shape)
        return {
            "seq_ids": sequence_ids,
            "seq_masks": sequence_masks,
            "target_list": batch_target, # need to add image embedding to it?
            "seq_imgs": sequence_imgs,
        }



class ItemDataset(Dataset):
    # TBD for eval & inference
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collect_fn(self, batch):
        # batch, batch_img = batch_tup[0], batch_tup[1] # images
        item_ids, item_masks, item_imgs = encode_batch(batch, self.tokenizer, self.args.item_size)
        # imgs = torch.cat(batch_img, dim=0)
        # imgs = imgs.unsqueeze(imgs, 1)
        return {
            "item_ids": item_ids,
            "item_masks": item_masks,
            "item_imgs": item_imgs
        }


def list_split(array,n):
    split_list = []
    s1 = array[:n]
    s2 = array[n:]
    split_list.append(s1)
    if len(s2) != 0:
        split_list.append(s2)
    return split_list




def encode_batch(batch, tokenizer, max_length):
    batch_text = [i[0] for i in batch]
    batch_img = torch.Tensor([i[1] for i in batch])
    padding = torch.zeros(batch_img.size(0), 256 - batch_img.size(1)) # to args later
    batch_img = torch.cat([batch_img, padding], dim=1)
    imgs = torch.unsqueeze(batch_img, 1) #[512, 1, 64+64]
    outputs = tokenizer.batch_encode_plus(
        batch_text,
        max_length=max_length,
        pad_to_max_length=True,
        return_tensors='pt',
        truncation=True,
    )
    input_ids = outputs["input_ids"]
    input_ids = torch.unsqueeze(input_ids, 1)
    attention_mask = outputs["attention_mask"]
    attention_mask = torch.unsqueeze(attention_mask, 1)
    return input_ids, attention_mask, imgs




def load_item_name(filename):
    # load name
    item_desc = dict()
    id_prefix = 'id:'
    title_prefix = 'title:'
    lines = open(filename, 'r').readlines()
    for line in lines[1:]:
        line = line.strip().split('\t')
        item_id = int(line[0])
        name = line[1]
        name = name.replace('&amp;', '')
        item_text = id_prefix + " " + str(item_id) + " " + title_prefix + " " + name
        item_desc[item_id] = item_text
    return item_desc


def load_item_address(filename):
    # load name and address
    item_desc = dict()
    id_prefix = 'id:'
    title_prefix = 'title:'
    passage_prefix = 'address:'
    lines = open(filename, 'r').readlines()
    for line in lines[1:]:
        line = line.strip().split('\t')
        item_id = int(line[0])
        name = line[1]
        address = line[3]
        city = line[4]
        state = line[5]
        item_text = id_prefix + " " + str(item_id) + " " + title_prefix + " " + name + " " + \
                    passage_prefix + " " + address + " " + city + " " + state
        item_desc[item_id] = item_text
    return item_desc
# load images
def load_item_image(filename):
    item_img = dict()
    embeddings = np.load(filename)
    for i in range(len(embeddings)):
        item_img[i+1]=embeddings[i].tolist()
    return item_img

def load_data(filename,item_desc,item_img):
    data = []
    lines = open(filename, 'r').readlines()
    for line in lines[1:]: # 100 samples first
        example = list()
        line = line.strip().split('\t')
        target = int(line[-1])
        seq_id = line[1:-1]
        text_list = []
        img_list = []
        for id in seq_id:
            id = int(id)
            if id==0:
                break
            text_list.append(item_desc[id])
            img_list.append(item_img[id])
        text_list.reverse()
        img_list.reverse() # img seq 
        seq_text = ', '.join(text_list)
        example.append(seq_text)
        example.append(target) # how about target image
        example.append(img_list)
        data.append(example)

    return data



def load_item_data(item_desc,item_img):
    data = []
    keys = item_desc.keys()
    for i in keys:
        text = item_desc[i]
        img = item_img[i]
        data.append((text, img)) # images

    return data