import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm

file = "unlabeled_data.csv"
df = pd.read_csv(file)
df['synonym_aug'] = 0
aug = naw.SynonymAug(aug_src='wordnet')
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    df['synonym_aug'][idx] = aug.augment(row['content'])
df.to_csv(file, index=None)
