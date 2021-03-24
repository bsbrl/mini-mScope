%% Video writing function to segment blue and green videos
%takes frame_color that contains the blue and green frame indices and uses
%them to write separate videos for calcium and hemodynamic channel for visual
%analysis, motion correction and other functions
%Daniel Surinach 03/20/2021
%%

clear;clc;close all;
%input folder directory with alternatingly pulsed LED video
input_param(1).foldername = 'O:\My Drive\BSBRL DATA REPOSITORY\PROJECTS\MESOSCOPE\Calcium&Behavior Camera Data For Sharing\SKylar intact skull\skylar_intact_mouse41_t1_Lvisual\merge\segmented_intactawakevisual_stim\';
vidlist = dir(fullfile(input_param(1).foldername,'*.avi')); %get .avi files outside main folder
final_list = struct('vidname',{},'keepordel',{});
for videoindex = 1:length(vidlist)
    frame = 1;
    input_param(1).vid_inputname = strcat(input_param(1).foldername,'\',vidlist(videoindex).name);%get full file name for video1
    rawvideo=VideoReader(input_param(1).vid_inputname);
    fprintf('Video name below\n')
    input_param(1).vid_inputname
    
    input_param(1).stopframe = rawvideo.Duration*rawvideo.FrameRate;    
    tempname=strsplit(input_param(1).vid_inputname,'.avi'); %split name from .avi for renaming analysis .mat files and video files
    input_param(1).blueorgreenonly{1} = 'segment';
    input_param(1).blueorgreenonly{2} = 'bg';
    
    %iterate through video and determine if there are any frame drops
    %correct frame drops with linear interpolation between consecutive
    %points and determine if the trial has sufficient stability to keep
    frame_color = {};
    [frame_color] = findframe(rawvideo,frame,input_param(1).stopframe,frame_color,0,0,strjoin(strcat(tempname(1),' mergedbluegreen.avi')),input_param(1).blueorgreenonly); %extract which frames are green and blue
    if isempty(frame_color(1).black)==1
        frame_color(1).black = 0;
    end
    
    %% save results from frame finding for behavior camera and other analysis
    %ensures that frame drops are accounted for in final analysis
    frame_color.blueoutput = unique(round(frame_color.blue/2));
    frame_color.greenoutput = unique(round(frame_color.green/2));
    save(strcat(tempname{1},'_frame_indices.mat'),'frame_color');
    
    
    %% generates a final list of trials to keep or delete
    %the user can view final_list variable and the remove the 
    %trials with too many frame drops or power surges from analysis
    final_list(videoindex).vidname = vidlist(videoindex).name;
    prompt = {'Keep or delete trial? (k/d)'};%let user choose if blue or green frame (manual segmentation)
    dlgtitle = 'Keep segmented trial';
    final_list(videoindex).keepordel = inputdlg(prompt,dlgtitle);
    
    %write the pulsing input video into calcium/hemo channel videos
    %that are separated for motion correction and more
    rawvideo=VideoReader(input_param(1).vid_inputname);
    v = VideoWriter(strcat(tempname{1},' bluevideo.avi'));
    v.FrameRate = rawvideo.FrameRate/2;
    open(v)
    v2 = VideoWriter(strcat(tempname{1},' greenvideo.avi'));
    v2.FrameRate = rawvideo.FrameRate/2;
    open(v2)
    %segment_video(frame_color,frame,rawvideo,v,v2,input_param(1).stopframe)
    
    blueindex = 1;
    greenindex = 1;


    wt = waitbar(0,'starting video analysis2');%progress bar to see how code processsteps=len;
    %stopframe = 3601;
    steps = input_param(1).stopframe;%total frames


    while hasFrame(rawvideo) %read through all the frames
        single_frame = readFrame(rawvideo); %grab current frame


        if blueindex <= length(frame_color(1).blue) %iterate through blue frame
            %indices and see if the current frame matches the desired index
            if frame == frame_color(1).blue(blueindex)
                writeVideo(v,single_frame)%if so, write the frame to the blue video file
                blueindex = blueindex+1;
            end
        end

        if greenindex <= length(frame_color(1).green)%repeat with green frames
            if frame == frame_color(1).green(greenindex)
                writeVideo(v2,single_frame)
                greenindex = greenindex+1;
            end

        end

        if frame==input_param(1).stopframe
            break
        end

        if mod(frame,20)==0
            waitbar(frame/steps,wt,sprintf('writing blue/green videos %1.0f/%1.0f',frame,steps))
        end

        frame = frame + 1;
    end
    close(wt)
    close(v)
    close(v2)
    
    
end


