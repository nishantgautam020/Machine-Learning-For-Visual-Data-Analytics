%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%coursework: face recognition with eigenfaces
%% 
% Path
addpath(genpath('software'));

%% 2.1 READING TRAINING AND TEST IMAGES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Loading of the images: You need to replace the directory 

Imagestrain = loadImagesInDirectory ( 'images/training-set/23x28/');  % Training Set
[Imagestest, Identity] = loadTestImagesInDirectory ( 'images/testing-set/23x28/');  % Testing Set


%% 2.2 CONSTRUCT MEAN IMAGE AND COVARIANCE MATRIX OF THE TRAINING SET
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ImagestrainSizes = size(Imagestrain);  %returns a row vector with elements of length equal to dimensions of Imagestrain

Means = floor(mean(Imagestrain)); %rounding off the mean of Imagestrain.

CenteredVectors = (Imagestrain - repmat(Means, ImagestrainSizes(1), 1));

CovarianceMatrix = cov(CenteredVectors); % Computing Covariance Matrix

% 2.3 COMPUTING EIGEN FACE

[U, S, V] = svd(CenteredVectors); % computing SINGULAR VALUE DECOMPOSITION
Space = V(: , 1 : ImagestrainSizes(1))'; 
Eigenvalues = diag(S); % Eigen Face

%% 2.4 MEAN IMAGE VISUALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MeanImage = uint8 (zeros(28, 23)); % Initializing and empty array of size 28*23
for k = 0:643
   MeanImage( mod (k,28)+1, floor(k/28)+1 ) = Means (1,k+1); 
 
end
figure;
subplot (1, 1, 1); %dividing current figure into an 1-by-1 grid and creates axes in the position specified by 1
imshow(MeanImage);
title('Mean Image');

%% 2.5 DISPLAY FIRST 20 EIGEN FACES 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% Code by ******** NISHANT GAUTAM (210832761) ***********

for j = 1:20
    eigen_faces = uint8 (zeros(28, 23)); % Initializing an empty array
    denorm_space = rescale(Space, 0,255) % 
    
    for m = 0:643
        eigen_faces( mod (m,28)+1, floor(m/28)+1 ) = denorm_space (j,m+1);
    end
    
    subplot (4, 5, j);
    imshow(eigen_faces);
    title(['Eigen Face',num2str(j)]);


end
% Code by ******** NISHANT GAUTAM (210832761) ***********



%% 2.6 PROJECTING TESTING AND TRAINING IMAGES ONTO THE 20 EIGEN IMAGES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Projection of the two sets of images omto the face space:

Locationstrain=projectImages (Imagestrain, Means, Space);
Locationstest=projectImages (Imagestest, Means, Space);

Threshold =20;

TrainSizes=size(Locationstrain);
TestSizes = size(Locationstest);
Distances=zeros(TestSizes(1),TrainSizes(1));

%% 2.7 COMPUTING THE DISTANCE BETWEEN PROJECT TEST IMAGES AND PROJECT TRAIN IMAGES


for i=1:TestSizes(1),
    for j=1: TrainSizes(1),
        Sum=0;
        for k=1: Threshold,
   Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end,
     Distances(i,j)=Sum;
    end,
end,

Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
for i=1:70,
[Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end,

%% 2.8 FIRST 6 RECOGNITION RESULT IMAGES VISUALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
x=6;
y=2;
for i=1:6,
      Image = uint8 (zeros(28, 23));
      for k = 0:643
     Image( mod (k,28)+1, floor(k/28)+1 ) = Imagestest (i,k+1);
      end,
   subplot (x,y,2*i-1);
    imshow (Image);
    title('Image tested');
    
    Imagerec = uint8 (zeros(28, 23));
      for k = 0:643
     Imagerec( mod (k,28)+1, floor(k/28)+1 ) = Imagestrain ((Indices(i,1)),k+1);
      end,
     subplot (x,y,2*i);
imshow (Imagerec);
title(['Image recognised with ', num2str(Threshold), ' eigenfaces:',num2str((Indices(i,1))) ]);
end,

%% 2.9 COMPUTING RECOGNITION RATE FOR TOP 20 EIGEN FACES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
averageRR=zeros(1,20);
for t=1:20,
  Threshold =t;  
Distances=zeros(TestSizes(1),TrainSizes(1));

for i=1:TestSizes(1),
    for j=1: TrainSizes(1),
        Sum=0;
        for k=1: Threshold,
   Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end,
     Distances(i,j)=Sum;
    end,
end,

Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
number_of_test_images=zeros(1,40);% Number of test images of one given person.%YY I modified here

for i=1:70,
number_of_test_images(1,Identity(1,i))= number_of_test_images(1,Identity(1,i))+1;%YY I modified here
[Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end,

rec_rate = [];
for i = 1: length(Imagestest(:,1))
    % if the indices of train does not match with Identity in test then rate is 0.
    if ceil(Indices(i,1)/5) == Identity(i)
        rec_rate(i) = 1;
    else 
        rec_rate(i) = 0;
    end
end
recognition_rate = sum(rec_rate)/70 *100;
averageRR(1,t) = recognition_rate;

end,
figure;
plot(averageRR(1,:));
title('Recognition rate against the number of eigenfaces used');
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

%% 2.10 INVESTIGATE THE EFFECT USING DIFFERENT NO. OF EIGEN FACES
%effect of threshold (i.e. number of eigenfaces):   

averageRR=zeros(1,20);
for t=1:60,
  Threshold =t;  
Distances=zeros(TestSizes(1),TrainSizes(1));

for i=1:TestSizes(1),
    for j=1: TrainSizes(1),
        Sum=0;
        for k=1: Threshold,
   Sum=Sum+((Locationstrain(j,k)-Locationstest(i,k)).^2);
        end,
     Distances(i,j)=Sum;
    end,
end,

Values=zeros(TestSizes(1),TrainSizes(1));
Indices=zeros(TestSizes(1),TrainSizes(1));
number_of_test_images=zeros(1,40);% Number of test images of one given person.%YY I modified here

for i=1:70,
number_of_test_images(1,Identity(1,i))= number_of_test_images(1,Identity(1,i))+1;%YY I modified here
[Values(i,:), Indices(i,:)] = sort(Distances(i,:));
end,

rec_rate = [];
for i = 1: TestSizes(1)
    % if the indices of train does not match with Identity in test then rate is 0.
    if ceil(Indices(i,1)/5) == Identity(i)
        rec_rate(i) = 1;
    else 
        rec_rate(i) = 0;
    end
end
recognition_rate = sum(rec_rate)/70 *100;
averageRR(1,t) = recognition_rate;

end,
figure;
plot(averageRR(1,:));
title('Recognition rate against the number of eigenfaces used');


%% 2.11 (a) Effect of K: Evaluating the effect of K in KNN 
%% 2.11 (b) Plotting the recognition rate against K, using 20 eigenfaces.

Locationstrain=projectImages (Imagestrain, Means, Space);
Locationstest=projectImages (Imagestest, Means, Space);
K = 100; % Equals to number of eigen faces
averageRR_K=zeros(1,K);

for k=1:K
     
    Distances=zeros(size(Locationstest,1), size(Locationstrain,1));
    Values = zeros(size(Locationstest,1), size(Locationstrain,1));
    Indices = zeros(size(Locationstest,1), size(Locationstrain,1));   
    
    for i=1:size(Locationstest,1)
        for j=1:size(Locationstrain,1)
            Distances(i,j) = sqrt(sum((Locationstest(i,:) - Locationstrain(j,:)) .^ 2));
        end
    end

    % Sorting viable matching training images for test images (selecting min distance -euclidean):
    [Values, Indices] = sort(Distances,2);
    index_matrix = floor((Indices(:,:)-1)/5)+1;

    % For each test sample check the recognition
     correct_count = 0; % initialize correct count for each K tested.
       

    for i = 1:size(Imagestest,1)
        knn_output = mode(index_matrix(i, 1:k));

        if (knn_output == Identity(i)) % The recognition is correct
            correct_count = correct_count + 1;
        end   
    end

    % Traversed through all samples, compute recognition rate:
    recog_rate_k = (correct_count/ size(Imagestest,1))*100;
    averageRR_K(1,k) = recog_rate_k;
    
end
% Plot the Recognition Rate:
figure;
plot(averageRR_K);
title('Recognition rate against K');
ylabel('Recognition Rate'); xlabel('K');