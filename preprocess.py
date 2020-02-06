from mydl.custom.protein.preprocess import combine_dataset, train_valid_split
from mydl.custom.protein.preprocess import count_distrib
from mydl.custom.protein.preprocess import create_sample_weight, create_class_weight

import os
import logging
logger = logging.getLogger("mydl")

def do_preprocess(path):
    a_csv = "HPAv18RBGY_wodpl.csv"
    b_csv = "HPAv18RGBY_WithoutUncertain_wodpl.csv"
    df_combined = combine_dataset(path, a_csv, b_csv)
    df_train, df_valid = train_valid_split(df_combined, 4)

    for i in range(4):    
        train_df = df_train[i]    
        train_df.to_csv(os.path.join(path, "train_split_%d.csv"%i), index=False)    
        valid_df = df_valid[i]    
        valid_df.to_csv(os.path.join(path, "valid_split_%d.csv"%i), index=False)

        create_sample_weight(path, train_df, 'train_split_%d_weights.pickle'%i)

    create_class_weight(path, dict(count_distrib(df_combined)), mu=0.5)


def main():
    do_preprocess('./datasets')

if __name__ == '__main__': main()
