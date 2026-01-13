%% Project Matrix and Tensor techniques for data science
clear; 
clc; 
close all;

addpath ./l1_ls_matlab

% Load the dataset:
load("dataset_ORL_56x46.mat");

% Save the variables of the dataset:
face_dataset = images;        % Matrix containing the 40*10 images vectorized (one image in each row)
label_dataset = labels;       % Row vector containing the index of the person


% Global parameters:
n_classes = 40;             % Number of classes/subjects
n_image_per_sub = 10;       % Total number of images per subject     
methods = {'OMP','l1_ls', 'FISTA'};        % Methods analyzed
n_runs = 10;                % Number of ripetitions for the experiment

% Parameter for the solvers:
tol = 1e-5;
maxit = 1000;

rng(42) % Fix the seed


%% EXPERIMENT 1: Analysis of the performance of the methods for different values of K and Lambda
% In particular we sub-divide this experiment in 2 parts: firstly we
% consider the original set of images without noise, then we add some
% white Gaussian noise and compare them.
fprintf('\nExperiment 1: Analysis of the methods with different values of K and Lambda. \n');

% Other parameters:
n_train = 5;        % We select 5 images for each subject to construct the training dataset
corruption_percent = 0.20; % 20% of the pixels are corrupted 

% Values of regularization parameter Lambda (to test):
%lambda_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100]; % Abbreviato per leggibilità
lambda_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10, 100];

% Values of sparsity parameter K (to test):
%k_values = [1, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100];                 
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 22, 25, 27, 30, 35, 40, 45, 50, 55];

% IMPORTANT: Use the same length for K and Lambda
if length(k_values)~=length(lambda_values)
    error('Check the number of values for K and lambda: They must coincide!');
end

% Matrix to save the all the values of accuracy: (Methods x Lambda_values/K_values)
all_accuracies = zeros(length(methods), length(lambda_values));
all_accuracies_noise_SP = zeros(length(methods), length(lambda_values));
all_accuracies_noise_WG = zeros(length(methods), length(lambda_values));

% Pre allocate the variables (We want different train/test at each run):
train_data = [];       
train_labels = [];
test_data = [];
test_labels = [];

% Random split for constructing the training and test dataset
for s = 1:n_classes
    index = find(label_dataset==s); % Indices of the images associated to the subj. s

    % Select training samples (First 5 images):
    index_train = index(1:n_train);  
    train_data = [train_data; face_dataset(index_train,:)];                                   
    train_labels = [train_labels; label_dataset(index_train)'];
    
    % Construct the test dataset (The remaining ones):
    index_test = index(n_train+1:end);
    test_data = [test_data; face_dataset(index_test,:)];                                   
    test_labels = [test_labels; label_dataset(index_test)'];

end

%% PCA (98%)
% Section IX-B: "principle component analysis algorithm is used... to preserve 98%% energy" 
fprintf('\nComputing PCA to preserve 98%% energy.\n');

% Centering the data wrt the mean of the training set
mean_face = mean(train_data, 1);

% Add Gaussian noise
sigma = corruption_percent;
noise_WG = (sigma*255) * randn(size(test_data));    % Since Y is not centered we need to multiply sigma*255 (to have a gaussina centered wrt our data = pixels in [0,255]
test_data_WG = test_data + noise_WG;

% Add noise to Y (Salt and pepper)
test_data_SP = test_data; 
[n_test_imgages, n_pixels] = size(test_data);

% Parameters for adding "salt and pepper"
val_salt = 255; % White
val_pepper = 0; % Black
n_corrupted_pixels = round(corruption_percent * n_pixels); 

for k = 1:n_test_imgages
    indices = randperm(n_pixels, n_corrupted_pixels);
    n_salt = round(n_corrupted_pixels / 2);
    
    % Salt
    test_data_SP(k, indices(1:n_salt)) = val_salt;
    % Pepper
    test_data_SP(k, indices(n_salt+1:end)) = val_pepper;
end

% Center the data (using the mean of D):
D_centered = train_data - mean_face;       
Y_centered = test_data - mean_face;  
Y_centered_SP = test_data_SP - mean_face; 
Y_centered_WG = test_data_WG - mean_face;

% PCA
S = cov(D_centered);    % Covariance matrix
[V, E] = eig(S);        % Eigenvector and eigenvalues
e = diag(E);            % Extract the vector of eigenvalues

% Order the eigenvalues (and corresponding eigenvectors)
[eig_sorted, I] = sort(e, 'descend');
V = V(:,I);

% Select the 98% of total energy
cum_energy = cumsum(eig_sorted) / sum(eig_sorted);
k_pca = find(cum_energy > 0.98, 1, 'first');  % take the k-first Principal component      

% Projection matrix (matrix A)
A = V(:,1:k_pca);

% Principal components
D = (D_centered * A)';

% Project Y using the PCA matrix A
Y_no_noise = (Y_centered * A)'; 

Y_noise_SP = (Y_centered_SP * A)';  % With SP noiseì
Y_noise_WG = (Y_centered_WG * A)';     % With WG noise

% CLASSIFIER: Algorithm 17 (Image classification)
% Step 1: Normalize all the samples to have unit l2-norm.
D = D./vecnorm(D);      % Each column is an image

Y = Y_no_noise./vecnorm(Y_no_noise);        % Original test dataset (No noise)
Y_noise_SP = Y_noise_SP./vecnorm(Y_noise_SP);        % Noisy test dataset
Y_noise_WG = Y_noise_WG./vecnorm(Y_noise_WG);        % Noisy test dataset

n_test_samples = size(Y, 2);
n_train_samples = size(D,2);    % Columns of the dictionary

% LOOP OVER THE METHODS
for m = 1:length(methods)
    method_name = methods{m};
    fprintf('\nExecuting method %s\n', method_name);

    % Loop over the lambdas/K
    for l = 1:length(lambda_values)
        K = k_values(l);
        lambda = lambda_values(l);

        % Selection of the solver
        switch method_name
            case 'OMP'
                X_hat = omp_solver(Y, D, K, tol);
                X_hat_noise_SP = omp_solver(Y_noise_SP, D, K, tol);
                X_hat_noise_WG = omp_solver(Y_noise_WG, D, K, tol);
            case 'l1_ls'                
                X_hat   = l1_ls_solver(Y, D, lambda, tol);
                X_hat_noise_SP = l1_ls_solver(Y_noise_SP, D, lambda, tol);
                X_hat_noise_WG = l1_ls_solver(Y_noise_WG, D, lambda, tol);
            case 'FISTA'
                X_hat   = FISTA_solver(Y, D, lambda, maxit, tol);
                X_hat_noise_SP   = FISTA_solver(Y_noise_SP, D, lambda, maxit, tol);
                X_hat_noise_WG   = FISTA_solver(Y_noise_WG, D, lambda, maxit, tol);

        end 
        
        % Classification (Residuals - Algorithm 17) 
        predicted_labels = zeros(n_test_samples, 1);            
        predicted_labels_noise_SP = zeros(n_test_samples, 1);
        predicted_labels_noise_WG = zeros(n_test_samples, 1);

        for j = 1:n_test_samples
            y = Y(:, j);     % Original test image
            x_hat = X_hat(:, j); % Sparse coefficient obtained with the solver
            
            % With noise:
            x_hat_noise_SP = X_hat_noise_SP(:, j); % Sparse coefficient obtained with the solver
            x_hat_noise_WG = X_hat_noise_WG(:, j); % Sparse coefficient obtained with the solver

            min_res = inf;
            best_class = -1;

            min_res_noise_SP = inf;
            best_class_noise_SP = -1;

            min_res_noise_WG = inf;
            best_class_noise_WG = -1;
            
            % Try to reconstruct the image using the coefficients of a
            % single class at each time
            for c = 1:n_classes
                % Find in the dictionary the columns that belong to the class 'c'
                idx_class = find(train_labels == c);

                % Construct a vector that contain ONLY the coefficients of class 'c'
                x_class_only = zeros(size(x_hat));
                x_class_only(idx_class) = x_hat(idx_class);
                
                x_class_only_noise_SP = zeros(size(x_hat_noise_SP));
                x_class_only_noise_SP(idx_class) = x_hat_noise_SP(idx_class);

                x_class_only_noise_WG = zeros(size(x_hat_noise_WG));
                x_class_only_noise_WG(idx_class) = x_hat_noise_WG(idx_class);

                % Reconstruct the image
                y_rec = D * x_class_only;
                y_rec_noise_SP = D * x_class_only_noise_SP;
                y_rec_noise_WG = D * x_class_only_noise_WG;

                % Compute the residual
                res = norm(y - y_rec)^2;
                res_noise_SP = norm(y - y_rec_noise_SP)^2;
                res_noise_WG = norm(y - y_rec_noise_WG)^2;
                
                % If the residual is the smallest found, that 'c' is
                % the class winning
                if res < min_res
                    min_res = res;
                    best_class = c;
                end
                
                if res_noise_SP < min_res_noise_SP
                    min_res_noise_SP = res_noise_SP;
                    best_class_noise_SP = c;
                end

                if res_noise_WG < min_res_noise_WG
                    min_res_noise_WG = res_noise_WG;
                    best_class_noise_WG = c;
                end
            end
            predicted_labels(j) = best_class;
            predicted_labels_noise_SP(j) = best_class_noise_SP;
            predicted_labels_noise_WG(j) = best_class_noise_WG;
        end
        
        % Compute accuracy            
        acc = sum(predicted_labels(:) == test_labels(:)) / n_test_samples;
        all_accuracies(m, l) = acc * 100;

        % Compute accuracy            
        acc_noise_SP = sum(predicted_labels_noise_SP(:) == test_labels(:)) / n_test_samples;
        all_accuracies_noise_SP(m, l) = acc_noise_SP * 100;

        % Compute accuracy            
        acc_noise_WG = sum(predicted_labels_noise_WG(:) == test_labels(:)) / n_test_samples;
        all_accuracies_noise_WG(m, l) = acc_noise_WG * 100;

        if strcmp(method_name, 'OMP')
            fprintf('Method: %s | K: %3d | Accuracy (no noise): %.2f %% | Accuracy (with noise SP): %.2f %% | Accuracy (with noise WG): %.2f %%\n', method_name, K, all_accuracies(m, l), all_accuracies_noise_SP(m, l),all_accuracies_noise_WG(m, l));
        else
            fprintf('Method: %s | Lambda: %7.4f | Accuracy (no noise): %.2f %% | Accuracy (with noise SP): %.2f %% | Accuracy (with noise WG): %.2f %% \n', method_name, lambda, all_accuracies(m, l), all_accuracies_noise_SP(m, l),all_accuracies_noise_WG(m, l));
        end
    end

end


%% For BOTH (NO NOISE AND WITH NOISE) 
% Save the best parameter and relative mean accuracy for each method
best_param = zeros(length(methods),1);
best_accuracy = zeros(length(methods), 1);
best_param_noise_SP = zeros(length(methods),1);
best_accuracy_noise_SP = zeros(length(methods), 1);
best_param_noise_WG = zeros(length(methods),1);
best_accuracy_noise_WG = zeros(length(methods), 1);

for m = 1:length(methods)   
    [max_acc, idx] = max(all_accuracies(m, :));
    [max_acc_n_SP, idx_n_SP] = max(all_accuracies_noise_SP(m, :));
    [max_acc_n_WG, idx_n_WG] = max(all_accuracies_noise_WG(m, :));
    best_accuracy(m) = max_acc;
    best_accuracy_noise_SP(m) = max_acc_n_SP;
    best_accuracy_noise_WG(m) = max_acc_n_WG;
   
    if strcmp(methods{m}, 'OMP')
        best_param(m) = k_values(idx);
        best_param_noise_SP(m) = k_values(idx_n_SP);
        best_param_noise_WG(m) = k_values(idx_n_WG);
    else
        best_param(m) = lambda_values(idx);
        best_param_noise_SP(m) = lambda_values(idx_n_SP);
        best_param_noise_WG(m) = lambda_values(idx_n_WG);
    end
end

data_table = [best_param, best_accuracy, best_param_noise_SP, best_accuracy_noise_SP,best_param_noise_WG, best_accuracy_noise_WG]';

% Define the names of the rows of the table:
row_names = {'Best parameter', 'Max accuracy', 'Best parameter (noise SP)', 'Max accuracy (noise SP)','Best parameter (noise WG)', 'Max accuracy (noise WG)'};

% Table
T = array2table(data_table, 'VariableNames', methods, 'RowNames', row_names);

fprintf('\nTable experiment 1:\n')
disp(T);

% Comparison plot (Fig. 2)
x_indices = 1:length(lambda_values);


figure(1);

% NO NOISE
subplot(2,3,1);
method = 1;         % OMP
plot(x_indices, all_accuracies(method, :), '-o','LineWidth', 2, 'MarkerSize', 6, 'DisplayName', methods{method})
xlabel('Sparsity parameter K');
ylabel('Accuracy (%)');
title(sprintf('Comparison (Avg accuracy vs K)'));
grid on;

% Formattazione asse X
ax = gca;
ax.XTick = x_indices;
ax.XTickLabel = string(k_values);
ax.XTickLabelRotation = 45;

% Legend
legend('Location', 'southwest');
ylim([84 98]); 
xlim([1, length(k_values)]);
% hold off;

subplot(2,3,4);
hold on;

% Remove OMP (start for m=2):
for m = 2:length(methods)
    plot(x_indices, all_accuracies(m, :), '-o','LineWidth', 2, 'MarkerSize', 6, 'DisplayName', methods{m});
end

xlabel('Regularization parameter \lambda');
ylabel('Accuracy (%)');
title(sprintf('Comparison (Accuracy vs Lambda)'));
grid on;

% Formattazione asse X
ax = gca;
ax.XTick = x_indices;
ax.XTickLabel = string(lambda_values);
ax.XTickLabelRotation = 45;

% Legend
legend('Location', 'southwest');
ylim([84 98]);
hold off;

% WITH NOISE SALT AND PEPPER:
%figure(2);

subplot(2,3,2);
method = 1;         % OMP
plot(x_indices, all_accuracies_noise_SP(method, :), '-o','LineWidth', 2, 'MarkerSize', 6, 'DisplayName', methods{method})
xlabel('Sparsity parameter K');
ylabel('Accuracy (%)');
title(sprintf('Comparison (Accuracy vs K) with noise Salt and Pepper'));
grid on;

% Formattazione asse X
ax = gca;
ax.XTick = x_indices;
ax.XTickLabel = string(k_values);
ax.XTickLabelRotation = 45;

% Legend
legend('Location', 'southwest');
ylim([84 98]);
xlim([1, length(k_values)]);
% hold off;

subplot(2,3,5);
hold on;

% Remove OMP (start for m=2):
for m = 2:length(methods)
    plot(x_indices, all_accuracies_noise_SP(m, :), '-o','LineWidth', 2, 'MarkerSize', 6, 'DisplayName', methods{m});
end

xlabel('Regularization parameter \lambda');
ylabel('Accuracy (%)');
title(sprintf('Comparison (Accuracy vs Lambda) with noise Salt and Pepper'));
grid on;

% Formattazione asse X
ax = gca;
ax.XTick = x_indices;
ax.XTickLabel = string(lambda_values);
ax.XTickLabelRotation = 45;

% Legend
legend('Location', 'southwest');
ylim([84 98]);
hold off;

% WITH NOISE WHITE GAUSSIAN:
%figure(3);

subplot(2,3,3);
method = 1;         % OMP
plot(x_indices, all_accuracies_noise_WG(method, :), '-o','LineWidth', 2, 'MarkerSize', 6, 'DisplayName', methods{method})
xlabel('Sparsity parameter K');
ylabel('Accuracy (%)');
title(sprintf('Comparison (Accuracy vs K) with noise Additive White Gaussian'));
grid on;

% Formattazione asse X
ax = gca;
ax.XTick = x_indices;
ax.XTickLabel = string(k_values);
ax.XTickLabelRotation = 45;

% Legend
legend('Location', 'southwest');
ylim([84 98]);
xlim([1, length(k_values)]);
% hold off;

subplot(2,3,6);
hold on;

% Remove OMP (start for m=2):
for m = 2:length(methods)
    plot(x_indices, all_accuracies_noise_WG(m, :), '-o','LineWidth', 2, 'MarkerSize', 6, 'DisplayName', methods{m});
end

xlabel('Regularization parameter \lambda');
ylabel('Accuracy (%)');
title(sprintf('Comparison (Accuracy vs Lambda) with noise Additive White Gaussian'));
grid on;

% Formattazione asse X
ax = gca;
ax.XTick = x_indices;
ax.XTickLabel = string(lambda_values);
ax.XTickLabelRotation = 45;

% Legend
legend('Location', 'southwest');
ylim([84 98]);
hold off;


% pause 

%% EXPERIMENT 2: Comparison of the methods with different size of train dataset
fprintf('\nExperiment 2: Analysis of the methods with different dimensions of train dataset. \n');

n_train_values = [1:6]; % Number of images per subject in the train dataset

% Initialize a variable to save the acuracy/time values for all the methods and
% for all different train dataset size
acc_values = zeros(length(methods), length(n_train_values), n_runs); 
time_values = zeros(length(methods), length(n_train_values), n_runs); 

for i = 1:length(n_train_values)
    n_train = n_train_values(i);

    fprintf('\n Training size: %d \n', n_train);
    for r = 1:n_runs
        fprintf('\n ---- RUN %d/%d ----\n', r, n_runs);
        % Pre allocate the variables (We want different train/test at each run):
        train_data = [];       
        train_labels = [];
        test_data = [];
        test_labels = [];
    
        % Random split for contructing the training and test dataset
        for s = 1:n_classes
            index = find(label_dataset==s); % Find the indices of the images contining subject s
            p = randperm(n_image_per_sub);  % Random permutation => to select 
                                            % randomly some of the images of person s
            
            % Select training samples:
            index_train = index(p(1:n_train));  % Apply the permutation to shuffle 
                                                % the images associated to the person 
                                                % s and select the first 5
            train_data = [train_data; face_dataset(index_train,:)];                                   
            train_labels = [train_labels; label_dataset(index_train)'];
            
             % Construct the test dataset (all the other images)
            index_test = index(p(n_train+1:end));
            test_data = [test_data; face_dataset(index_test,:)];                                   
            test_labels = [test_labels; label_dataset(index_test)'];
        end
   
        
        %% PCA (98%)
        % Section IX-B: "principle component analysis algorithm is used... to preserve 98%% energy" 
        % fprintf('Computing PCA to preserve 98%% energy.\n');
    
        % Centering the data wrt the mean of the training set
        mean_face = mean(train_data, 1);
        D_centered = train_data - mean_face;
        Y_centered = test_data - mean_face; 
        
        % PCA
        S = cov(D_centered);    % Covariance matrix
        [V, E] = eig(S);        % Eigenvector and eigenvalues
        e = diag(E);            % Extract the vector of eigenvalues
    
        % Order the eigenvalues (and corresponding eigenvectors)
        [eig_sorted, I] = sort(e, 'descend');
        V = V(:,I);
        
        % Select the 98% of total energy
        cum_energy = cumsum(eig_sorted) / sum(eig_sorted);
        k_pca = find(cum_energy > 0.98, 1, 'first');  % take the k-first Principal component      
        
        % Projection matrix (matrix A)
        A = V(:,1:k_pca);
        
        % Principal components
        D_final = (D_centered * A)';
        Y_final = (Y_centered * A)'; 
    
        % CLASSIFIER: Algorithm 17 (Image classification)
        % Step 1: Normalize all the samples to have unit l2-norm.
        D = D_final./vecnorm(D_final);      % Each column is an image
        Y = Y_final./vecnorm(Y_final);
        
        n_test_samples = size(Y, 2);
        n_train_samples = size(D,2);    % Columns of the dictionary
    
        % LOOP OVER THE METHODS
        for m = 1:length(methods)
            
            method_name = methods{m};
            %fprintf('\nExecuting method %s\n', method_name);
            parameter = best_param(m);     % Save the best paramter (K or Lambda) found at Experim. 1
    
            tic;
            switch method_name
                case 'OMP'
                    X_hat = omp_solver(Y, D, parameter, tol);
                case 'l1_ls'      
                    % X_train = l1_ls_solver(D, D, lambda, tol);
                    X_hat   = l1_ls_solver(Y, D, parameter, tol);
                case 'FISTA'
                    % X_train = FISTA_solver(D, D, lambda, maxit, tol);
                    X_hat   = FISTA_solver(Y, D, parameter, maxit, tol);
            end 
               
           
            % Classification (Residuals - Algorithm 17) 
            predicted_labels = zeros(n_test_samples, 1);            
            
            for j = 1:n_test_samples
                y = Y(:, j);     % Original test image
                x_hat = X_hat(:, j); % Sparse coefficient obtained with the solver
                
                min_res = inf;
                best_class = -1;
                
                % Try to reconstruct the image using the coefficients of a
                % single class at each time
                for c = 1:n_classes
                    % Find in the dictionary the columns that belong to the class 'c'
                    idx_class = find(train_labels == c);
                    
                    % Construct a vector that contain ONLY the coefficients of class 'c'
                    x_class_only = zeros(size(x_hat));
                    x_class_only(idx_class) = x_hat(idx_class);
                    
                    % Reconstruct the image
                    y_reconstructed = D * x_class_only;
                    
                    % Compute the residual
                    res = norm(y - y_reconstructed)^2;
                    
                    % If the residual is the smallest found, that 'c' is
                    % the class winning
                    if res < min_res
                        min_res = res;
                        best_class = c;
                    end
                end
                predicted_labels(j) = best_class;
            end
            time_values(m,i,r) = toc;
            % Compute accuracy            
            acc_values(m,i,r) = (sum(predicted_labels(:) == test_labels(:)) / n_test_samples)*100;
        end 
    end
end

acc_mean = mean(acc_values,3);
time_mean = mean(mean(time_values,3),2);     % Average time over all the runs and over all the different n_train size                                           

% Name of the rows
names = cell(1, length(n_train_values) + 1);
for i = 1:length(n_train_values)
    names{i} = ['Train Size = ', num2str(n_train_values(i))]; 
end
names{end} = 'Average time';

% Matrix of the values:
table_matrix = [acc_mean'; time_mean'];

% Table:
T2 = array2table(table_matrix, 'VariableNames', methods, 'RowNames', names);

fprintf('\n Table experiment 2:\n');
disp(T2);

% pause

%% Experiment 3: Comparison of the methods with a noisy dataset
fprintf('\nExperiment 3: Analysis of the methods with different levels of noise.\n');

noise_levels = [0, 0.1, 0.2, 0.3]; % Percentage of corrupted pixels/ Different sigmas

acc_noise_SP = zeros(length(methods), length(noise_levels), n_runs);
acc_noise_WG = zeros(length(methods), length(noise_levels), n_runs);

% Fix the number of train dataset to compare the methods (we use the same
% as Experiment 1 = 5)
n_train_noise = 5; 

for n = 1:length(noise_levels)
    sigma = noise_levels(n);
    fprintf('\n Noise level = %.2f \n', sigma);
    
    for r = 1:n_runs
        fprintf('\n ---- RUN %d/%d ----\n', r, n_runs);

        % Split Train/Test dataset (As before)
        train_data = []; 
        train_labels = []; 
        test_data = []; 
        test_labels = [];

        for s = 1:n_classes
            index = find(label_dataset==s);
            p = randperm(n_image_per_sub);
            train_data = [train_data; face_dataset(index(p(1:n_train_noise)),:)];
            train_labels = [train_labels; label_dataset(index(p(1:n_train_noise)))'];

            test_data = [test_data; face_dataset(index(p(n_train_noise+1:end)),:)];
            test_labels = [test_labels; label_dataset(index(p(n_train_noise+1:end)))'];
        end

        % Center the data:
        mean_face = mean(train_data, 1);
        
        % Add Gaussian noise
        noise_WG = (sigma*255) * randn(size(test_data));    % Since Y is not centered we need to multiply sigma*255 (to have a gaussina centered wrt our data = pixels in [0,255]
        test_data_WG = test_data + noise_WG;
        
        % Add noise to Y (Salt and pepper)
        test_data_SP = test_data; 
        [n_test_imgages, n_pixels] = size(test_data);
        
        % Parameters for adding "salt and pepper"
        val_salt = 255; % White
        val_pepper = 0; % Black
        n_corrupted_pixels = round(sigma * n_pixels); 
        
        for k = 1:n_test_imgages
            indices = randperm(n_pixels, n_corrupted_pixels);
            n_salt = round(n_corrupted_pixels / 2);
            
            % Salt_
            test_data_SP(k, indices(1:n_salt)) = val_salt;
            % Pepper
            test_data_SP(k, indices(n_salt+1:end)) = val_pepper;
        end
        
        % Center the data (using the mean of D):
        D_centered = train_data - mean_face;       
        Y_centered = test_data - mean_face;  
        Y_centered_SP = test_data_SP - mean_face; 
        Y_centered_WG = test_data_WG - mean_face;

        % PCA (98%)
        S = cov(D_centered); 
        [V, E] = eig(S);
        [eig_sorted, I] = sort(diag(E), 'descend');
        V = V(:,I);
        cum_energy = cumsum(eig_sorted) / sum(eig_sorted);
        
        k_pca = find(cum_energy > 0.98, 1, 'first');
        A = V(:,1:k_pca);
        
        D = (D_centered * A)'; 

        % Project the noise Y
        Y_SP = (Y_centered_SP * A)';     % With SP noise
        Y_WG = (Y_centered_WG * A)';     % With WG noise

        % Final normalization:
        D = D ./ vecnorm(D);
        Y_SP = Y_SP ./ vecnorm(Y_SP);
        Y_WG = Y_WG ./ vecnorm(Y_WG);
        
        n_test_samples = size(Y_SP, 2);

        % Apply the methods
        for m = 1:length(methods)
            method_name = methods{m};
            % fprintf('\nExecuting method %s\n', method_name);

            parameter_SP = best_param_noise_SP(m);
            parameter_WG = best_param_noise_WG(m);
 
            switch method_name
                case 'OMP'
                    X_hat_SP = omp_solver(Y_SP, D, parameter_SP, tol);
                    X_hat_WG = omp_solver(Y_WG, D, parameter_WG, tol);
                case 'l1_ls'
                    X_hat_SP = l1_ls_solver(Y_SP, D, parameter_SP, tol);
                    X_hat_WG = l1_ls_solver(Y_WG, D, parameter_WG, tol);
                case 'FISTA'
                    X_hat_SP = FISTA_solver(Y_SP, D, parameter_SP, maxit, tol);
                    X_hat_WG = FISTA_solver(Y_WG, D, parameter_WG, maxit, tol);
            end
            
            % Classification (Residuals - Algorithm 17) 
            % predicted_labels = zeros(n_test_samples, 1);
            predicted_labels_SP = zeros(n_test_samples, 1);
            predicted_labels_WG = zeros(n_test_samples, 1);
            
            for j = 1:n_test_samples
                y_SP = Y_SP(:, j);     % Original test image
                y_WG = Y_WG(:, j);     % Original test image
                x_hat_SP = X_hat_SP(:, j); % Sparse coefficient obtained with the solver
                x_hat_WG = X_hat_WG(:, j); % Sparse coefficient obtained with the solver
                
                min_res_SP = inf;
                best_class_SP = -1;
                min_res_WG = inf;
                best_class_WG = -1;
                
                % Try to reconstruct the image using the coefficients of a
                % single class at each time
                
                for c = 1:n_classes
                    % Find in the dictionary the columns that belong to the class 'c'
                    idx_class = find(train_labels == c);
                    
                    % Construct a vector that contain ONLY the coefficients of class 'c'
                    x_class_only_SP= zeros(size(x_hat_SP));
                    x_class_only_SP(idx_class) = x_hat_SP(idx_class);
                    x_class_only_WG= zeros(size(x_hat_WG));
                    x_class_only_WG(idx_class) = x_hat_WG(idx_class);
                    
                    % Reconstruct the image
                    y_reconstructed_SP = D * x_class_only_SP;
                    y_reconstructed_WG = D * x_class_only_WG;
                    
                    % Compute the residual
                    res_SP = norm(y_SP - y_reconstructed_SP)^2;
                    res_WG = norm(y_WG - y_reconstructed_WG)^2;
                    
                    % If the residual is the smallest found, that 'c' is
                    % the class winning
                    if res_SP < min_res_SP
                        min_res_SP = res_SP;
                        best_class_SP = c;
                    end
                    if res_WG < min_res_WG
                        min_res_WG = res_WG;
                        best_class_WG = c;
                    end
                end
                predicted_labels_SP(j) = best_class_SP;
                predicted_labels_WG(j) = best_class_WG;
            end
            acc_noise_SP(m, n, r) = (sum(predicted_labels_SP(:) == test_labels(:)) / n_test_samples) * 100;
            acc_noise_WG(m, n, r) = (sum(predicted_labels_WG(:) == test_labels(:)) / n_test_samples) * 100;
        end
    end
end

% Mean accuracy (over the 5 runs)
mean_acc_noise_SP = mean(acc_noise_SP, 3);
mean_acc_noise_WG = mean(acc_noise_WG, 3);
%%
% Final plot
figure(4);
subplot(1,2,1)
plot(noise_levels, mean_acc_noise_SP(1,:)', '-o', 'LineWidth', 2,'MarkerSize', 8, 'DisplayName', 'OMP'); % Linea spessa
hold on;
plot(noise_levels, mean_acc_noise_SP(2,:)', '-o', 'LineWidth', 2,'MarkerSize', 8, 'DisplayName', 'l1_ls'); % Linea spessa
hold on;
plot(noise_levels, mean_acc_noise_SP(3,:)', '--o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'FISTA'); % Linea tratteggiata e più sottile
% plot(noise_levels, mean_acc_noise_SP', '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Level of noise (%)');
ylabel('Accuracy (%)');
title('Robustness to noise Salt and Pepper');
legend(methods, 'Location', 'SouthWest');
grid on;

%figure(5);
subplot(1,2,2)
plot(noise_levels, mean_acc_noise_WG(1,:)', '-o', 'LineWidth', 2,'MarkerSize', 8, 'DisplayName', 'OMP'); % Linea spessa
hold on;
plot(noise_levels, mean_acc_noise_WG(2,:)', '-o', 'LineWidth', 2,'MarkerSize', 8, 'DisplayName', 'l1_ls'); % Linea spessa
hold on;
plot(noise_levels, mean_acc_noise_WG(3,:)', '--o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'FISTA'); % Linea tratteggiata e più sottile
% plot(noise_levels, mean_acc_noise_WG', '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Level of noise (\sigma)');
ylabel('Accuracy (%)');
title('Robustness to noise Additive White Gaussian');
legend(methods, 'Location', 'SouthWest');
grid on;


