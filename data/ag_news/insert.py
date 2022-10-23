import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm


file = "unlabeled_data.csv"
df = pd.read_csv(file)
df['insert_aug'] = 0
aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert")
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    df['insert_aug'][idx] = aug.augment(row['content'])
df.to_csv(file, index=None)
