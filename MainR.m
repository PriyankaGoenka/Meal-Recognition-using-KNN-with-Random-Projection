% created by Priyanka Goenka
clc
clear
close all
M=readmatrix('DatasetReduced.csv');
%M= M(randperm(size(M, 1)), :);
%save('DatasetReduced.mat','M'); %Conversion from csv file to mat file
ytrain=M(:,end);
traindata=M(:,1:end-1);

testdata=readmatrix('ReleasedDatasetReduced.csv');
ytest=testdata(:,end);
testdata=testdata(:,1:end-1);

ypredicted=zeros(size(testdata,1),1);
 %numCores = feature('numcores');
 %p = parpool(numCores);
k=1500;% Positive interger (neighbor). Value of k will vary in kNN classifier. For different values of k we get different results. 
tic
parfor i = 1:size(testdata,1)
    ypredicted(i)=knn(testdata(i,:),traindata,ytrain,k);%KNN classification
end
confumat=confusionmat(ytest,ypredicted);
accuracy=sum(diag(confumat))/size(testdata,1)*100%misclassified/total*100
toc