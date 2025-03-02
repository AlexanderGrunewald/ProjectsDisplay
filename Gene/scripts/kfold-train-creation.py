# create a kfold train and test set for a ensemble blending model
from sklearn.model_selection import KFold
import pandas as pd

if __name__ == "__main__":
    # read in the training data
    df = pd.read_csv("../input/train.csv")

    # create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df.target.values

    # initiate the kfold class from model_selection module
    kf = KFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # save the new csv with kfold column
    df.to_csv("../input/train_folds.csv", index=False)
