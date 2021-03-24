%% This file reconcatenates motion corrected calcium/hemodynamic video
%Saves it in the original format where calcium and hemodynamic videos
%pulse alternately. This will be used for the hemodynamic corrected seed
%pixel maps used in the spontaneous_behavior function
%Daniel Surinach March 2021
%%
close all;clc;clear;
%input directory folder that contains separated calcium/reflectance videos
folder = 'H:\spontaneous behavior\9-17-20_trial1\segmented_spontaneous_videos\rigid\';

vid_files = dir(fullfile(folder,'*.avi'));
mat_data = dir(fullfile(folder,'*.mat'));
mat_data = load(strcat(folder,mat_data.name));

first_frame = mat_data.first_frame_color;

if strcmp(first_frame,'b') == 1 %if first frame was blue, concatenate video using blue first
    first_color = 'bluevideo';
else
    first_color = 'greenvideo';
end

vidname_split = strsplit(vid_files(1).name,{' ','.'}); %split video name to see if bluevideo or greenvideo
vidname_split = vidname_split{end-1};
if strcmp(vidname_split,first_color) == 1%initialize output video
    v = VideoReader(strcat(folder,vid_files(1).name));
    v2 = VideoReader(strcat(folder,vid_files(2).name));
    
else
    v = VideoReader(strcat(folder,vid_files(2).name));
    v2 = VideoReader(strcat(folder,vid_files(1).name));

end

vidname_split = strsplit(vid_files(1).name,first_color);

v3 = VideoWriter(strcat(folder,vidname_split{1},'.avi'));
v3.FrameRate = 2*v.FrameRate;%double framerate of each video
open(v3)

max_frame = min(v.Duration*v.Framerate,v2.Duration*v2.Framerate);

wt=waitbar(0,'Extracting Frames');%progress bar to see how code processsteps=len;
steps=round(v.Duration*v.Framerate);%total frames
k = 0;
while hasFrame(v) %read frames from both videos
    k = k+1;
    if k <= max_frame %write finalized video to alternately pulse
        im1 = readFrame(v);
        im2 = readFrame(v2);
        
        writeVideo(v3,im1);
        writeVideo(v3,im2);
    else
        break
    end

   if mod(k,20)==0
        waitbar(k/steps,wt,sprintf('Extracting frame data for frame %1.0f/%1.0f',k,steps))
   end
    
end
close(v3)
close(wt)