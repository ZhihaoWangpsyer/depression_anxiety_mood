clear;clc;close all;
load('health_data.mat')

% choice modelling
for isub=1:2%length(healthy_data)
    for istart=1:50
        for imodel=1:3                      
        % fit the data
            inx{1,1} = 10*rand;    lb{1,1} = 0;      ub{1,1} = 10; % EV model
            inx{1,2} = [0.5+4.5*rand rand+0.3 rand+0.3 10*rand];  lb{1,2} = [0.5 0.3 0.3 0];      ub{1,2} = [5 1.3 1.3 10]; % Prospect model
            inx{1,3} = [0.5+4.5*rand rand+0.3 rand+0.3 10*rand 2*rand-1 2*rand-1];  lb{1,3} = [0.5 0.3 0.3 0 -1 -1];      ub{1,3} = [5 1.3 1.3 10 1 1]; % Prospect model
            if imodel==1
                fitoutput_tmp{istart,imodel}  = fit_evmodel(inx{1,imodel},lb{1,imodel},ub{1,imodel},healthy_data{isub,1});
            elseif imodel==2
                fitoutput_tmp{istart,imodel}  = fit_prospectmodel(inx{1,imodel},lb{1,imodel},ub{1,imodel},healthy_data{isub,1});
            elseif imodel==3
                fitoutput_tmp{istart,imodel}  = fit_aamodel(inx{1,imodel},lb{1,imodel},ub{1,imodel},healthy_data{isub,1});
            end
            clear inx lb ub
            LL(istart,imodel) = fitoutput_tmp{istart,imodel}.modelLL;
        end
    end
    for imodel=1:3
        temp    = find(LL(:,imodel)==max(LL(:,imodel)));
        indexbest(imodel) = temp(1);
        fitoutput_choice{isub,imodel} = fitoutput_tmp{indexbest(imodel),imodel}; 
        clear temp
    end
    clear fitoutput_tmp LL indexbest
end
clear isub istart imodel

% genreation aamodel latent variables
for isub=1:2%length(healthy_data)
    fitoutput_choice{isub,3} = generation_aamodel(fitoutput_choice{isub,3},healthy_data{isub,1});
end
%save choice_model_fit.mat

choice_model = fitoutput_choice;
data_all = healthy_data;
% Mood modeling
n_model=17;
for isub=1:2%length(file_names) % remove ppt 41 due to 90 choices only (but not 150)
    for istart=1:50
        for imodel=1:n_model
            % M1:t; M2:CR*t; M3:EV*t; M4:RPE*t; M5: Win-Loss; M6: Win-Loss+t;
            % M7: no CR; M8: no EV; M9: no RPE; M10: no CR+t; M11: no EV+t; M12: no RPE+t;
            % M13: subjective + t
              inx{1,1} = [4*rand-2 4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,1} = [-2 -2 -2 0 -2 -2];   ub{1,1}=[2 2 2 1 2 2];
              inx{1,2} = [4*rand-2 4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,2} = [-2 -2 -2 0 -2 -2];   ub{1,2}=[2 2 2 1 2 2];
              inx{1,3} = [4*rand-2 4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,3} = [-2 -2 -2 0 -2 -2];   ub{1,3}=[2 2 2 1 2 2];
              inx{1,4} = [4*rand-2 4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,4} = [-2 -2 -2 0 -2 -2];   ub{1,4}=[2 2 2 1 2 2];
              inx{1,5} = [4*rand-2 4*rand-2 rand 4*rand-2];       lb{1,5} = [-2 -2 0 -2];   ub{1,5}=[2 2 1 2];
              inx{1,6} = [4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,6} = [-2 -2 0 -2 -2];   ub{1,6}=[2 2 1 2 2];
              inx{1,7} = [4*rand-2 4*rand-2 rand 4*rand-2];       lb{1,7} = [-2 -2 0 -2];   ub{1,7}=[2 2 1 2];
              inx{1,8} = [4*rand-2 4*rand-2 rand 4*rand-2];       lb{1,8} = [-2 -2 0 -2];   ub{1,8}=[2 2 1 2];
              inx{1,9} = [4*rand-2 4*rand-2 rand 4*rand-2];       lb{1,9} = [-2 -2 0 -2];   ub{1,9}=[2 2 1 2];
              inx{1,10} = [4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,10} = [-2 -2 0 -2 -2];   ub{1,10}=[2 2 1 2 2];
              inx{1,11} = [4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,11} = [-2 -2 0 -2 -2];   ub{1,11}=[2 2 1 2 2];
              inx{1,12} = [4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,12} = [-2 -2 0 -2 -2];   ub{1,12}=[2 2 1 2 2];
              inx{1,13} = [4*rand-2 4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,13} = [-2 -2 -2 0 -2 -2];   ub{1,13}=[2 2 2 1 2 2];
            % m1:M2+t
            % m2:M3+t
            % m3:M4+t
            % m4:M5+t
            
              inx{1,14} = [4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,14} = [-2 -2 0 -2 -2];   ub{1,14}=[2 2 1 2 2];
              inx{1,15} = [4*rand-2 4*rand-2 4*rand-2 rand rand rand 4*rand-2 4*rand-2];       lb{1,15} = [-2 -2 -2 0 0 0 -2 -2];   ub{1,15}=[2 2 2 1 1 1 2 2];
              inx{1,16} = [4*rand-2 4*rand-2 4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,16} = [-2 -2 -2 -2 0 -2 -2];   ub{1,16}=[2 2 2 2 1 2 2];
              inx{1,17} = [4*rand-2 4*rand-2 4*rand-2 4*rand-2 rand 4*rand-2 4*rand-2];       lb{1,17} = [-2 -2 -2 -2 0 -2 -2];   ub{1,17}=[2 2 2 2 1 2 2];


              if imodel==1
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_1(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==2
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_2(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==3
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_3(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==4
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_4(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==5
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_5(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==6
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_6(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==7
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_7(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==8
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_8(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==9
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_9(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==10
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_10(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==11
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_11(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==12
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_12(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==13
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw_13(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==14
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw2_t(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==15
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw3_t(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==16
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw4_t(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              elseif imodel==17
                 fitoutput_tmp{istart,imodel}  = fit_happiness_raw5_t(inx{1,imodel},lb{1,imodel},ub{1,imodel},data_all{isub,1},choice_model{isub,3});
              end
              clear inx lb ub
              LL(istart,imodel) = fitoutput_tmp{istart,imodel}.r2;
        end
    end

    for imodel=1:n_model
        temp    = find(LL(:,imodel)==max(LL(:,imodel)));
        indexbest(imodel) = temp(1);
        mood_model{isub,imodel} = fitoutput_tmp{indexbest(imodel),imodel}; 
        clear temp
    end
    clear fitoutput_tmp LL indexbest
end
%save mood_model_fit.mat