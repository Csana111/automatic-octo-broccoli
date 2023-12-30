import sweetviz as sweetviz
import pandas as pd

train = pd.read_csv('pc_X_train.csv')
test = pd.read_csv('pc_X_test.csv')
sv = sweetviz.analyze([train, "Train"], pairwise_analysis='off')
sv.show_html('EDA.html')
