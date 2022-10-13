from fast_ml.model_development import train_valid_test_split
import pandas as pd

df = pd.read_csv('fitzpatrick17k.csv', low_memory=False)


X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df, target='fitzpatrick_scale',
                                                                            train_size=0.8, valid_size=0.1, test_size=0.1,
                                                                            random_state=42)

# Join the X and y dataframes back together
X_train["fitzpatrick_scale"] = y_train
X_valid["fitzpatrick_scale"] = y_valid
X_test["fitzpatrick_scale"] = y_test

# output the results to csv files
X_train.to_csv("fitz17k_train_random_holdout.csv", index=False)
X_valid.to_csv("fitz17k_val_random_holdout.csv", index=False)
X_test.to_csv("fitz17k_test_random_holdout.csv", index=False)