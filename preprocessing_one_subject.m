function [target_trial_x_interp, target_trial_y_interp] = preprocessing_one_subject(EyeX_,EyeY_,Misses,Response_Validity,RTs,Targ_Location,Target_CoordinatesX, Target_CoordinatesY, ~, list)

% Benedetta Franceschiello, Lausanne, May 2018
% Fabio Anselmi, IIT and MIT, Genova
% Eye - Tracker project: Analysis of ET trajectories for Neglect
% identification with Signal Processing techniques

% This function takes into account RAW data, it preprocesses them and gives
% as output the normalized fft transform + normalized Raw Data

%% Select trials based on the experimental design (trials with Misses == 0 means the participant reached the target, then we consider Valid and Invalid trials)

tot_target = 8; %total number of targets in the experiment (both right and left)
index_trials = 1:size(EyeX_,1);
%Include Valid and Invalid trials // 1 trial == 1 trajectory 
idx_to_be_kept = index_trials((Misses == 0)&((Response_Validity == 1) | (Response_Validity == 0)))';

assert(sum(Misses(idx_to_be_kept))==0)

% Clean up the dataset // keep trials according to the above-mentioned
% experimental conditions
EyeX_ = EyeX_(idx_to_be_kept,:);
EyeY_ = EyeY_(idx_to_be_kept,:);
Targ_Location = Targ_Location(idx_to_be_kept);
RTs=RTs(idx_to_be_kept); %RTs are in ms // RT = Reaction Times
% Index of end of the trial need to be rounded, despite the sampling rate
% reaction time starts at 3000 ms, so after first 1000 points of the vector
idx_react_time = round(RTs./3) + 1000; % RTs starts from 3000 ms , so we sum (1000 indexes) after converting
%ms in indexes

clear Misses; clear Response_Validity; clear index_trials
%% Inizialization and Outlier identification : screen resolution = 1025 x 768
%  NB: Discard target = 9 (we don't have target coordinates) 

target_present = unique(Targ_Location); %We verify the total number of targets within the recordings. This corresponds to the target effectievly present within the experiment.
%Initialise variables
target_index = cell(1,tot_target);
target_trial_x = cell(1,tot_target);
target_trial_y = cell(1,tot_target);
idx_react_time_per_target = cell(1,tot_target);

%loop over each target
for i = 1:tot_target
    
    %if the target is recorded during the experiment
    if ~isempty(find(target_present,i))
        
        target_index{i} = find(Targ_Location == i); %identify indexes corresponding to a target
        target_trial_x{i} = EyeX_(target_index{1,i},:);   %identify [Trial x rows] corresponding to a target
        target_trial_y{i} = EyeY_(target_index{1,i},:);
        idx_react_time_per_target{i} = idx_react_time(target_index{1,i}); %RTs starts from 3000 ms
        
    end
    
end

%% Preprocessing
% 1. Fill the gaps (NaN) with interpolated points
% NB: trajectories X and Y are handled separately. 

indexes = 1:size(target_trial_x{i},2); % i is not relevant, cause here we want the length of the trial, i.e. 3000
%initialise variables
target_trial_x_interp = cell(1,tot_target);
target_trial_y_interp = cell(1,tot_target);

%loop over target
for i = 1:length(list)
    
    %if the target is recorded during the experiment, i.e. we have trials
    %corresponding to that target
    if (find(target_present == list(i)) ~= 0)
        
        %chose one target
        one_target = list(i);
        
        %take all trials of that target
        for j=1:size(target_trial_x{one_target},1)
            
            % if first point of the sequence is NaN, replace with the center of
            % the screen to allow interpolation. Screen resolution = 1025 x 768
            if (isnan(target_trial_x{one_target}(j,1))==1 || isnan(target_trial_y{one_target}(j,1))==1)
                target_trial_x{one_target}(j,1) = 384;
                target_trial_y{one_target}(j,1) = 512;
            end
            
            % Filling last part of the trial (trajectory) with target coordinate after target identification (MOUSE pressed based on RT)
            
            if  idx_react_time_per_target{one_target}(j,1) <= size(target_trial_x{one_target},2)
                target_trial_x{one_target}(j,idx_react_time_per_target{one_target}(j,1):end) = Target_CoordinatesX(one_target);
                target_trial_y{one_target}(j,idx_react_time_per_target{one_target}(j,1):end) = Target_CoordinatesY(one_target);
            end
            
            %if zeros are found, put a NaN so we can interpolate
            target_trial_x{one_target}(j,target_trial_x{one_target}(j,:)==0) = NaN;
            target_trial_y{one_target}(j,target_trial_y{one_target}(j,:)==0) = NaN;
            assert(sum(target_trial_x{one_target}(j,:) == 0) == 0)
            assert(sum(target_trial_y{one_target}(j,:) == 0) == 0)
            
            %identify query points for interpolation, and fx at query points
            idx_not_Nan = double(indexes(isnan(target_trial_x{one_target}(j,:))==0));
            f_idx_not_Nan = double(target_trial_x{one_target}(j,isnan(target_trial_x{one_target}(j,:))==0));
            idy_not_Nan = double(indexes(isnan(target_trial_y{one_target}(j,:))==0));
            f_idy_not_Nan = double(target_trial_y{one_target}(j,isnan(target_trial_y{one_target}(j,:))==0));
            
            %Interpolation nearest neighbour 
            f_idx_Nan = interp1(idx_not_Nan,f_idx_not_Nan,indexes,'nearest');
            f_idy_Nan = interp1(idy_not_Nan,f_idy_not_Nan,indexes,'nearest');            
            
            %store interpolated trial
            target_trial_x_interp{one_target}(j,:) = f_idx_Nan;
            target_trial_y_interp{one_target}(j,:) = f_idy_Nan;
            
            %Test to check if there are NaN left after interpolation
            assert(sum(find(isnan(target_trial_x_interp{one_target}(j,:))))==0)
            assert(sum(find(isnan(target_trial_y_interp{one_target}(j,:))))==0)
            
            clear idx_not_Nan; clear idy_not_Nan; clear f_idx_Nan; clear f_idy_Nan
              
        end

        assert(length(target_trial_x_interp{one_target}(1,1:end-1))~= length(target_trial_x_interp{one_target}(1,1:idx_react_time_per_target{one_target}(1,1))))
    end
end
end
