clear;clc;close all;
%% video joiner function call
%reads miniscope videos in their stored folder and concatenates into one
%video of the same length as experiment duration

mesocam_folder = 'E:\buzzstim_long\H17_M18_S26_m3_rhl\';%original root with miniscope videos (keep \ at the end)
output_folder = 'E:\abf files\m3\H17_M18_S26_m3_rhl\';%where you want to store a copy of the merged video for data analysis (keep \ at the end)
meso_camnum = 0;
beh_camnum = 1;
meso_framerate = 30;
beh_framerate = 30;

video_joiner(mesocam_folder,output_folder,meso_camnum,beh_camnum,meso_framerate,beh_framerate);
close all;
%% video segmentation by stimulus excel time stamp
%segment full length video by smaller stimulus experiment time stamps
%stored by you as a user that set the experiment duration and stimulus time
%the file should be saved as a .xlsx as provided in a template which
%contains LED warmup time stamps, stimulus on/off time stamp, experiment
%end time stamps for each stimulus 
output_folder = stimulus_exp_segment_longtrials(output_folder);

%% blue/green frame video segmentation 
%segments each invididual trial above into respective blue/green videos
%here you can see if any frame drops (drastic drops in blue or green
%intensity) are present and you can make note of which trials to discard
segment_video(output_folder)
close all;

%% DF/F calculation blue channel
%peform DF/F for each video in the blue channel and then average all the
%DF/F into one single video
frac = 1;     %fraction of video for Fo in DF/F (i.e. 0.5 = half video used for Fo)
fil = 2;      % spatial filtering pixel radius (integer), LP used 2
fscale = 2;   % filtering weight (larger number = greater weight)
bin = 0.1;     % binning fraction (0 to 1, 1 = no binning)
t_saveframe = [4,7]; %time scale to save avg dff video to show heat map evolution

stim_time = 5;%stimulus experiment details
stim_duration = 1;
analysis_folder = strcat(output_folder,'\blue\');

stimulus_experiments(frac,fil,fscale,bin,analysis_folder,t_saveframe,stim_time,stim_duration)

%% DF/F calculation green channel
%peform DF/F for each video in the green channel and then average all the
%DF/F into one single video
analysis_folder = strcat(output_folder,'\green\');

stimulus_experiments(frac,fil,fscale,bin,analysis_folder,t_saveframe,stim_time,stim_duration)

%% hemodynamic correction
%hemodynamic correction using the average DF/F blue and green videos above
tstart = 1;
tend = 'full'; %Fo value in time
analysis_folder = strcat(output_folder,'\hemo\');
analysis_type = 'hemo_corr_avgdff';
stimulus_exp_hemocorr(bin,t_saveframe,stim_time,stim_duration,analysis_folder,analysis_type)

close all


