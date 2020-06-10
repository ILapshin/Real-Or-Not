import pandas as pd 
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt 

def output(data, file_name, file_path='data/sample_submission.csv'):
    # Creates a .csv file ready for Kaggle submission
    sample_submission = pd.read_csv(file_path)
    sample_submission['target'] = data
    sample_submission.to_csv(file_name + '.csv', index=False)

def verify(y, y_head):
    # Returns a fraction of right predictions of original training data set
    print(metrics.accuracy_score(y, y_head))

def plot(arr, res):
	# Squeezes an arr array to 2D and plots it scattered with color depending on res
    
    try:
        arr = PCA(n_components=2).fit_transform(arr)
    except Exception as e:
        arr = TruncatedSVD(n_components=2).fit_transform(arr)

    for i, item in enumerate(arr):
        color = 'g'
        if res[i] == 1:
            color = 'r'
        plt.scatter(item[0], item[1], c = color)
    plt.show()

