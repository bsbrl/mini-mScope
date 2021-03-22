%% Video writing function to segment blue and green videos
%takes frame_color that contains the blue and green frame indices and uses
%them to write separate videos for blue and green frames for visual
%analysis 

function segment_video(input_folder)

input_param(1).foldername = input_folder;
vidlist = dir(fullfile(input_param(1).foldername,'*.avi')); %get .avi files outside main folder
mkdir(strcat(input_param(1).foldername,'\','blue'));
mkdir(strcat(input_param(1).foldername,'\','green'));
for videoindex = 1:length(vidlist)
    frame = 1;
    vidname = vidlist(videoindex).name;
    input_param(1).vid_inputname = strcat(input_param(1).foldername,'\',vidname);%get full file name for video1
    rawvideo=VideoReader(input_param(1).vid_inputname);
    fprintf('Video name below\n')
    input_param(1).vid_inputname
    
    input_param(1).stopframe = rawvideo.Duration*rawvideo.FrameRate;    
    tempname=strsplit(input_param(1).vid_inputname,'.avi'); %split name from .avi for renaming analysis .mat files and video files
    input_param(1).blueorgreenonly{1} = 'segment';
    input_param(1).blueorgreenonly{2} = 'bg';
    
    
    frame_color = {};
    [frame_color] = findframe(rawvideo,frame,input_param(1).stopframe,frame_color,0,0,strjoin(strcat(tempname(1),' mergedbluegreen.avi')),input_param(1).blueorgreenonly); %extract which frames are green and blue
    if isempty(frame_color(1).black)==1
        frame_color(1).black = 0;
    end
    
    
    rawvideo=VideoReader(input_param(1).vid_inputname);
    vidname = strsplit(vidname,'.avi');
    vidname = vidname{1};
    vblue = VideoWriter(strcat(input_folder,'\blue\',vidname,' bluevideo.avi'));
    vblue.FrameRate = rawvideo.FrameRate/2;
    open(vblue)
    
    vgreen = VideoWriter(strcat(input_folder,'\green\',vidname,' greenvideo.avi'));
    vgreen.FrameRate = rawvideo.FrameRate/2;
    open(vgreen)
    %segment_video(frame_color,frame,rawvideo,v,v2,input_param(1).stopframe)
    
    blueindex = 1;
    greenindex = 1;

    %rawvideo=VideoReader(input_param(1).vid_inputname);
    %v=VideoWriter('blue.avi');
    %open(v)

    wt = waitbar(0,'starting video analysis2');%progress bar to see how code processsteps=len;
    %stopframe = 3601;
    steps = input_param(1).stopframe;%total frames


    while hasFrame(rawvideo) %read through all the frames
        single_frame = readFrame(rawvideo); %grab current frame


        if blueindex <= length(frame_color(1).blue) %iterate through blue frame
            %indices and see if the current frame matches the desired index
            if frame == frame_color(1).blue(blueindex)
                writeVideo(vblue,single_frame)%if so, write the frame to the blue video file
                blueindex = blueindex+1;
            end
        end

        if greenindex <= length(frame_color(1).green)%repeat with green frames
            if frame == frame_color(1).green(greenindex)
                writeVideo(vgreen,single_frame)
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
    close(vblue)
    close(vgreen)
    
    
end
end

