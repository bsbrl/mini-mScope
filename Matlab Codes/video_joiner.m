% Daniel Surinach
% concatenate miniscope separated videos into one long video

function video_joiner(mesocam_folder,output_folder,meso_camnum,beh_camnum,meso_framerate,beh_framerate)
% prompt1={'Enter desired video length to compress videos into (in seconds)'};
% dlgtitle1='Vid length';
% setLength=str2double(strjoin(inputdlg(prompt1,dlgtitle1,[1 50])));

[final_frames,setLength] = frame_pacing(mesocam_folder,meso_camnum,beh_camnum,meso_framerate,beh_framerate); %get common frames between meso and beh cam
first_frame_color = 'b';
save(strcat(mesocam_folder,'meso_beh_commonframes.mat'),'final_frames','first_frame_color','-v7.3'); %save list of frames common between meso and beh that are time stamped accurately

%%
int_list = {'0','1','2','3','4','5','6','7','8','9'};
temp = dir(strcat(mesocam_folder,'*.avi'));        % all .avi video clips
videoList = {temp.name}';   % sort this list if necessary! sort(videoList) might work
for j = 1:length(videoList)
    if j == 1
        %%
        [name,splits] = strsplit(videoList{j},{'.avi','_',' '});
        diff_char = find(strcmp(name,strsplit(videoList{j+1},{'.avi','_',' '}))==0); %find diff char between names
        if length(splits) > 1
            root_name1 = strcat(name(1),splits(1));
            root_name2 = strcat(splits(diff_char),name(diff_char+1));
            cond = 1;
            for k = 2:length(splits)
                if cond == 1
                    root_name1 = strcat(root_name1,name(k),splits(k));%recreate original video string
                end
                if k == diff_char-1%find where video names are different
                    cond = 2;
                elseif k >= diff_char+1
                   root_name2 = strcat(root_name2,splits(k),name(k+1)); 
                end

            end
        else
            name = name{1};
            for k = 1:length(name) %check where current name has int in it
                int_name = strcmp(name(k),int_list);
                if any(int_name) == 1 %if found integer 
                    int_name = int_list(int_name);
                    int_name = int_name{1};
                    root_name1 = strsplit(name,int_name);
                    root_name1 = root_name1{1};
                end
            end
           root_name2 = splits(1);
        end
        
    end
    app_name = strcat(root_name1,num2str(j),root_name2);
    sorted_list{j} = app_name{1};
    
end

sorted_list = sorted_list';
videoList = sorted_list;

fprintf('\nCheck video files have been sorted by name below...')
videoList

%%
outputVideo = VideoWriter(strcat(mesocam_folder,'mergedVideo2.avi'));
outputVideo.Quality=100;

% if all clips are from the same source/have the same specifications, and
% are the desired values for the output video, just initialize framerate
% with the settings of the first video in videoList:
inputVideo_init = VideoReader(strcat(mesocam_folder,videoList{1})); % first video
outputVideo.FrameRate = inputVideo_init.FrameRate;

   
open(outputVideo) % >> open stream
% iterate over all videos you want to merge (e.g. in videoList)
for i = 1:length(videoList)
    % select i-th clip (assumes they are in order in this list!)
    inputVideo = VideoReader(strcat(mesocam_folder,videoList{i}));
    % -- stream your inputVideo into an outputVideo
    while hasFrame(inputVideo)
        writeVideo(outputVideo, readFrame(inputVideo));
    end
    fprintf('Finished reading video %1.0f out of %1.0f\n',i,length(videoList))
end
close(outputVideo) % << close after having iterated through all videos



% Add code that allows the video length to be resized.  This is done by 
% manipulating the frame rate

fprintf('\nWriting video with %4.2f second length\n',setLength')
vidObj = VideoReader(strcat(mesocam_folder,'mergedVideo2.avi'));
numFrames = 0;
while hasFrame(vidObj)
     readFrame(vidObj);
     numFrames = numFrames + 1;
end

outputname = strsplit(mesocam_folder,'\');
outputVideo2 = VideoWriter(strcat(mesocam_folder,outputname{end-1},'.avi'));
outputVideo2.Quality=100;
outputVideo2.FrameRate = numFrames/setLength;

open(outputVideo2) % >> open stream
% iterate over all videos you want to merge (e.g. in videoList)

% select i-th clip (assumes they are in order in this list!)
inputVideo2 = VideoReader(strcat(mesocam_folder,'mergedVideo2.avi'));
% -- stream your inputVideo into an outputVideo
intensity = struct('totalframe',{});
blackcount = 1;
frame = 1;
intensity(1).blackind(1,1) = 0;
while hasFrame(inputVideo2)
    single_frame = readFrame(inputVideo2); %grab current frame

    if size(single_frame,3)==3
        single_framegraylarge=rgb2gray(single_frame); %convert rgb to grayscale
    else
        single_framegraylarge = single_frame;
    end

    intensity(1).totalframe(frame,1) = mean(mean(single_framegraylarge));

    if intensity(1).totalframe(frame) < 5  %if looking at mostly black frame
        intensity(1).black(blackcount,1) = intensity.totalframe(frame);
        intensity(1).blackind(blackcount,1) = frame;
        
        if isempty(find(diff(intensity(1).blackind)~=1)) == 1 %consecutive black frames
            fprintf('Consecutive Black frames detected at frame %1.0f\n',frame)
            final_consecblackind = blackcount;
            blackcount = blackcount+1;
        end
        
    else
        if intensity(1).blackind == 0
            final_consecblackind = 0;
        end
        %final_consecblackind = blackcount-1; 
        if frame >= final_consecblackind+3 %skip first 3 frames that are usually whack
            if frame == final_consecblackind+3
                imshow(single_frame)
                prompt = {'Is this a blue or green frame or neither? (b/g/n)'};%let user choose if blue or green frame (manual segmentation)
                dlgtitle = 'Blue and green frame selection';
                user_input = inputdlg(prompt,dlgtitle);
                %totalframes = rawvideo.Duration*rawvideo.FrameRate;
                if strcmp(user_input{1},'n') == 0 %if blue or green frame (exclude behavior cam concat prompt for this)
                    first_frame_color = user_input{1};
                    save(strcat(mesocam_folder,'meso_beh_commonframes.mat'),'final_frames','first_frame_color','-v7.3'); %save list of frames common between meso and beh that are time stamped accurately
                end
            end
            writeVideo(outputVideo2, single_frame);
%             if isempty(find(frame == final_frames,1)) == 0 %if frame in final_frames, write it to video
%                 writeVideo(outputVideo2, single_frame);
%             end
        end
    end

    frame = frame+1;
        
end
    
close(outputVideo2) % << close after having iterated through all videos
close(outputVideo)
copyfile(strcat(outputVideo2.Path,'\',outputVideo2.Filename),output_folder); %copy to analysis folder

% The following section of code can be used to count the exact number of
% frames in a video.  Just replace "myfile.avi" with the actual name of 
% the video (it must be in the current folder.
%
% vidObj = VideoReader('myfile.avi');
%     numFrames = 0;
%     while hasFrame(vidObj)
%         readFrame(vidObj);
%         numFrames = numFrames + 1;
%     end
end
