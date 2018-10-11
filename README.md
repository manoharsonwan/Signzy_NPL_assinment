# Signzy_NPL_assinment
#importing the useful library
import pandas as pd
#importing a data set as tsv file by using pandas library 
Train_dataset = pd.read_csv('labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
# cleaning the data by importing the "re" library 
# remove all other charecter than word like numbers, question marks, dots ect. And apply for loop so it can clean all data
review = re.sub('[^a-zA-Z]', ' ', train_dataset['review'][i])
#makes all word in lowes case
review = review.lower()
#taking the root of the word like loved to love
from nltk.stem.porter import PorterStemmer
corpus = [] #ampty list of all review
for i in range(0, 25000):
    review = re.sub('[^a-zA-Z]', ' ', train_dataset['review'][i]) # it remove all other chatecter than word like number, dots etc.
    review = review.lower() # makes all word in lowes case
    review = review.split() # it split all word seperatly 
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review) #the join allclean word with a single space 
    corpus.append(review) # makes corpus of all clean review
# creating a bag of words model this mwthod help us to create a unique column for each words through this a matrix are created for each word and then creating matrix of feature for dependent variable which in 0 or 1 and independent variable is review. tis helps to create a classification 
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer # create a sparse metrix for each word seperatly 
cv = CountVectorizer(max_features = 2000)# taking only 2000 words wich is meaningful for review response we take more than this if cpu can handle 
X = cv.fit_transform(corpus).toarray() # create a array to use in classification model 
y = train_dataset.iloc[:, 1].values # extractin a dependend variable which is positive or negative review as 0 and 1
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split #spliting a training set and test set of independent and dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) # take 20% test size to tainr the data 
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB #from the sklearn imroiting GaussianNB beause Naive Beyes theorem only took on parameter to obeserve and classify the model
classifier = GaussianNB()
classifier.fit(X_train, y_train)# fitting the test and training set to the Naive bayes theorem
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # making the confusion matrix to predict the accuracy of model

Through this model i took 5000 data set inn test set where in found 3680 correct prediction so my accuracy was 73.6%


