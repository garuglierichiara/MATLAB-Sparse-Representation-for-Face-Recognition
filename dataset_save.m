%% Preparation of the dataset DATASET:
clear; clc; close all;

% Folder which contains all the images of the dataset:
dataset_folder = 'archive';

% Name of the filw where to save the correct dataset:
output_filename = 'dataset_ORL_56x46.mat'; 

n_subjects = 40;       % Number of subjects/classes of the dataset
n_img_per_subj = 10;   % Number of images per subjects
dim_new = [56, 46];    % Dimension of the images specified in the paper


% Compute the dimension of the vectorized image: 56 * 46 = 2576 pixel
n_pixels = dim_new(1) * dim_new(2);

% Pre-allocation of the matrices where to save the images (vectorized) and
% the labels:
images = zeros(n_subjects * n_img_per_subj, n_pixels);
labels = zeros(1, n_subjects * n_img_per_subj);

% Loop to save the dataset:
row = 0; % Index for scrolling the columns and saving the images in a matrix

% Loop over the subjects (folders)
% In 'archive' there are 40 sub-folders: s1, ..., s40 and in each
% sub-folder there are the 10 images of each correspondent subject
for i = 1:n_subjects
    % Save correctly the name of the sub-folder s1, s2, ...
    subject_folder = fullfile(dataset_folder, sprintf('s%d', i));  % Find the path for the subfolder 
                                                                   % (in this way it works in every PC) 
    % Loop over all the images for a given subj.
    for j = 1:n_img_per_subj
        row = row + 1;
        
        % Path for the file of the images .pgm
        img_name = fullfile(subject_folder, sprintf('%d.pgm', j));
       

        % If the file exists:
        if isfile(img_name)
            % Read the image
            img = imread(img_name);
            
            % Resize the images
            image_resized = imresize(img, dim_new);
            
            % Vectorization (to save the images as rows)
            image_vec = double(image_resized(:)');
            
            % Salvataggio nella matrice
            images(row, :) = image_vec;
            labels(row) = i;
        else
            error('File does not exist: subj %d - image %d',i,j);
        end
    end
end

%% Save the dataset:

% Save the 3 variables: images labels and dim_new
save(output_filename, 'images', 'labels', 'dim_new');    
