from sklearn import svm

def trainClassifier(features, labels):
    clf = svm.SVC()
    clf.fit(features, labels)
    return clf

def classify(classifier, feature_vector):
    class_ = classifier.predict(feature_vector)
    return class_
