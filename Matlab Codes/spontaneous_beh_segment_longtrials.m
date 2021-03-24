%% separate LED warmuptime from videos
%This function uses a .xlsx file to separate LED warmup time
%from all videos for spontaneous behavior trials
%Daniel Surinach, 03/20/2021


%%
clear;clc;
foldername = 'E:\stimululs experiments\paper revision stim\11-3-2020_intact_awakevisualstim_trial5_highpower_lowgainexposure\blue_fixindividually\';
%input folder directory that contains videos and LED warmup time stamp

exptype = 'segmented_spontaneous_videos';%experiment type
mkdir(strcat(foldername,'\',exptype))%makes output directory
vidlist = dir(fullfile(foldername,'*.avi'));%get video and excel files
xlslist = dir(fullfile(foldername,'*.xlsx'));


for i = 1:length(vidlist) %iterate through videos in list (i.e. usually mesoscope and behavior cam)
    led_warmup_count = 1;
    vidlist = dir(fullfile(foldername,'*.avi'));
    xlslist = dir(fullfile(foldername,'*.xlsx'));

    txtdata = xlsread(strcat(foldername,'\',xlslist(1).name)); 
    timerows = 1:3:size(txtdata,1);
    timestamps = txtdata(timerows);
    led_warmup_time = timestamps(1)/1000; %time stamp of led warm up in sec
    exp_end_times = timestamps(3:2:end)/1000;%time stamp for trial end in sec
    
    vidname = strsplit(vidlist(i).name,'.avi');
    vidname = vidname{1};
    
    rawvideo=VideoReader(strcat(foldername,'\',vidlist(i).name));
    time_increment = 1/rawvideo.FrameRate; %video step in time in seconds
    
    wt = waitbar(0,'starting video analysis2');%progress bar to see how code processsteps=len;
    steps = rawvideo.FrameRate*rawvideo.Duration;%total frames
    frame = 1;
    tolerance = 0.025; %0.01 sec tolerance to get frame where LED warmuptime was
    vidcount = 1; %how many videos are written
    
    %store the end frames for each experiment section
    %this will determien where the video is split and into how many parts
    vidtime = ((1/rawvideo.FrameRate):(1/rawvideo.FrameRate):rawvideo.Duration)';
    for j = 1:length(exp_end_times)
       index = find(vidtime >= exp_end_times(j),1);
       if isempty(index) == 0 % if not at last time instance 
            exp_end_frame(j) = index;
       else
           exp_end_frame(j) = steps;
       end
       %[vidtime(a),exp_end_times(j)] 
    end
    
    %% read video and split LED warmup time and trial times if included
    while hasFrame(rawvideo)
       single_frame = readFrame(rawvideo); %grab current frame 
       
       if led_warmup_count == 1 %begin output video writing after LED warmuptime
           if abs(led_warmup_time - rawvideo.CurrentTime) <= tolerance
               fprintf('Reached LED warmup time at %4.4fsec\n',led_warmup_time)
               v = VideoWriter(strcat(foldername,'\',exptype,'\',vidname,'_',num2str(vidcount),' segment.avi'));
               v.FrameRate = rawvideo.FrameRate;
               open(v)
               vidcount = vidcount + 1;
               led_warmup_count = 2;
           end
       end

       if vidcount > 1 %if several trials to separate in time stamp xlsx file
           %create a new video for each individual trial for analysis later
           writeVideo(v,single_frame)
           
           if vidcount - 1 < length(exp_end_times)
               if frame == exp_end_frame(vidcount-1)%abs(exp_end_times(vidcount - 1) - rawvideo.CurrentTime) <= tolerance
                  fprintf('End of video segmentation for trial %1.0f\n',vidcount - 1)
                  close(v)

                  v = VideoWriter(strcat(foldername,'\',exptype,'\',vidname,'_',num2str(vidcount),' segment.avi'));
                  v.FrameRate = rawvideo.FrameRate;
                  open(v)
                  vidcount = vidcount + 1;
               end
           end
       end
              
       if mod(frame,20)==0
            waitbar(frame/steps,wt,sprintf('auto segmenting video %1.0f frame %1.0f/%1.0f',i,frame,steps))
       end
       
       vidtime(frame) = rawvideo.CurrentTime;
       frame = frame + 1;
        
    end
    close(wt)
    fprintf('End of video segmentation for trial %1.0f\n',vidcount - 1)
    close(v)
end