import  numpy as np
from operator import itemgetter
import csv,os
class KNNClassifier(object):
    def __init__(self):
        self.training_features=None
        self.training_labels=None
        self.test_features=None
        #build meangful result
        self.elegantResult="Most likely ,{0}'{1}'is of type of '{2}',"
    def loadTrainingDataFromFile(self,file_path):
        if  file_path is not None and os.path.exists(file_path):
            tr_features=[]
            self.training_labels=[]
            with open(file_path,'r')as training_data_file:
                reader=csv.DictReader(training_data_file)
                for row in reader:
                    if row['moviename']!='?':
                        tr_features.append([float(row['kicks']),float(row['kisses'])])
                        self.training_labels.append(row['movietypes'])
                    else:
                         self.test_features=np.array([float(row['kicks']),float(row['kisses'])])
            if len(tr_features)>0:
                self.training_features=np.array(tr_features)
            print('self.training_features:\n',self.training_features)
            print('self.training_labels:',self.training_labels)
            print('self.test_features:',self.test_features)
            #[1,1,1]
            #[18,90]
            #k=5
    def classifyTestData(self,test_data=None,k=0):
        print('classifyTestData:test data:',test_data)
        if test_data is not None:
            self.test_features=np.array(test_data,dtype=float)
        print('classifyTestData:self.test_features:',self.test_features)
        #ensure we have training data,training labels,test data
        if self.test_features is not None and self.training_features is not None\
                                  and self.training_labels is not None and k>0:
            print('classifyTest data says self.test_features:',self.test_features)
            print('self.training_feature:\n',self.training_features)
            print('self.training_labels:',self.training_labels)
            featureVectorSize=self.training_features.shape[0]
            print('featureVectorSize:',featureVectorSize)
            tileOfTestData=np.tile(self.test_features,(featureVectorSize,1))
            print('after ttileOfTestdata:\n',tileOfTestData)
            difMat=self.test_features-tileOfTestData
            print('diffMat:\n',difMat)
            sqDiffMat=difMat**2
            print('sqDiffMat:\n',sqDiffMat)
            sqDistances=sqDiffMat.sum(axis=1)
            print('(row wise sum)sqDistance:',sqDistances)
            distances=sqDistances**0.5
            print('distance (squre root of sqDistance): ',distances)
            sortedDistanceIndices=distances.argsort()
            print('sortedDistanceIndices:',sortedDistanceIndices)
            print('self.training_labels',self.training_labels)
            classCount={}
            for i in range(k):
                print('sortDistanceIndices[',i,']:',sortedDistanceIndices)
                voteILabel=self.training_labels[sortedDistanceIndices]
                print('voteILabel:',voteILabel)
                classCount[voteILabel]=classCount.get(voteILabel,0)+1
                #classCount={'action:2,'romance':3}
            print('classCount=',classCount)
            sortedClassCount=sorted(classCount.items(),key=itemgetter(1),reverse=True)
            #sorted class count ={'romance':3,'action':2}
            print('sortedClassCount=',sortedClassCount)
            print('sortedClassCount[0]:',sortedClassCount[0])
            print('sortedclasscount[0][0]:',sortedClassCount[0][0])
            return sortedClassCount[0][0]
        else:
            return 'cant determine result  from the empty test data'
    def predictMovieType:
        instance=KNNClassifier()
        instance.loadTrainingDataFromFile('LgR_Movies_kNN_classifier.csv')
        print("***********************************")
        my_test_data=[50,50]
        classOfTestData=instance.classifyTestData(test_data=my_test_data,k=5)
        return instance.elegantResult.format('movie',str(instance.test_features),classOfTestData)
    if __name__=='__main__':
         print(predictMovieType())







