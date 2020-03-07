from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load IMDB dataset
print('Loading dataset')

# Information on this function: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html#sklearn.datasets.load_files
imdbDataset = load_files(
    container_path='../data/imdb',
    #categories=['pos-subset','neg-subset'] This is just for quick testing, as loading the entire dataset is very slow
    categories=['pos','neg'],
    load_content=True,
    shuffle=True,
    random_state=42)

print('Loading complete')

# Split the dataset in training and test set
# Note that the features are still not in matrix form
# Information on this function: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=test_split#sklearn.model_selection.train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(
    imdbDataset.data,
    imdbDataset.target,
    test_size=0.5)

# Convert feature file data into sparse matrices
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(docs_train)
X_test_counts = count_vect.fit_transform(docs_train)

#Start using the data from here.
print(X_train_counts.shape)
print(y_train.shape)
print(X_test_counts.shape)
print(y_test.shape)