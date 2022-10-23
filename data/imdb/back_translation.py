# import torch
# from tqdm.notebook import tqdm
# import pandas as pd


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     n_gpu = torch.cuda.device_count()
#     print("gpu num: ", n_gpu)
#     en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
#     de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

#     en2de = en2de.cuda()
#     de2en = de2en.cuda()
#     file = "train_200.csv"
#     df = pd.read_csv(file)
#     # df.columns = ['label', 'content']
#     df['back_translation'] = 0
#     temperature = 0.9
#     for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
#         df['back_translation'][idx] = de2en.translate(en2de.translate(row['content'],  sampling = True, temperature = temperature),  sampling = True, temperature = temperature)
#     df = df.drop(['content', 'synonym_aug'], axis=1)
#     df.to_csv("bt_200.csv", index=None)

import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)
file = "unlabeled_data.csv"
df = pd.read_csv(file)
df['back_translation'] = 0
back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de', 
    to_model_name='facebook/wmt19-de-en',
    device=device
)
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    df['back_translation'][idx] = back_translation_aug.augment(row['content'])
# df = df.drop(['synonym_aug', 'content'], axis=1)
df.to_csv(file, index=None)