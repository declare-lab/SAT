import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm


file = "unlabeled_data.csv"
df = pd.read_csv(file)
df['random_delete'] = 0
aug = naw.RandomWordAug(aug_p=0.1)
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    df['random_delete'][idx] = aug.augment(row['content'])
df.to_csv(file, index=None)
