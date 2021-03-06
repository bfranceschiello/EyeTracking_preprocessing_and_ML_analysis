%% load data
clc
clear; close all;
addpath Dataset

%% load and pre-process all the file within a Folder

mat = dir('Dataset/*.mat');
X_all = [];
Y_all = [];
Y_label = [];
% Choose the list you want to use for saving the trajectories/images after
% preprocessing corresponding to the left sided targets / right sided
% targets or all of them.
lista = [3,4,5,6]; % left_pathological side
%lista2 = [1,2,7,8]; % right_side target
%list_all = [1,2,3,4,5,6,7,8]; %all target

%loop over subjects
for q = 1:length(mat)
    
    load(mat(q).name)
    length_traject = size(EyeX_,2);
    EyeX_unif = zeros(size(EyeX_,1),3000);
    EyeY_unif = zeros(size(EyeY_,1),3000);
    
    % Filling in until 9s (length = 3000, sampling rate 0.003), uniforming RAW data 
    EyeX_unif(:,(length_traject+1):3000) = NaN;
    EyeX_unif(:,1:length_traject) = EyeX_;
    EyeY_unif(:,(length_traject+1):3000) = NaN;
    EyeY_unif(:,1:length_traject) = EyeY_;
    
    % Call a function that computes the mat images for the Neural network
    [rawx,rawy] = preprocessing_one_subject(EyeX_unif,EyeY_unif,Misses,Response_Validity,RTs,Targ_Location,Target_CoordinatesX, Target_CoordinatesY,q,lista);
    
        for i=1:length(lista)
            %Create X raw matrix, with Label vector; 
            X_all = [X_all; rawx{lista(i)}];
            Y_all = [Y_all; rawy{lista(i)}];
            Y_label = [Y_label; [Label*ones(size(rawx{lista(i)},1),1), lista(i)*ones(size(rawx{lista(i)},1),1), q*ones(size(rawx{lista(i)},1),1)]];
        end
    
    clear rawx; clear EyeX_; clear EyeY_; clear EyeX_unif; clear EyeY_unif; 
    clear Label; clear Misses; clear Response_Validity; clear RTs;
end

%% Store label variables for ML analyses 
Y_p = Y_label(:,1);
ID_Tr = Y_label(:,3);
% Here we could save X_all, Y_all, Y_p and ID_Tr but it's not needed here
% as we already provide those in the main folder.