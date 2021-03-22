
function output_folder = stimulus_exp_segment_longtrials(input_folder)
% foldername = 'E:\abf files\m3\H17_M18_S26_m3_rhl';
%exptype = 'segmented_ledstim_videos'; %stimulus exp type
exptype = 'segmented_spontaneous_videos';
mkdir(strcat(input_folder,'\',exptype))
vidlist = dir(fullfile(input_folder,'*.avi'));

for i = 1:length(vidlist)
    vidlist = dir(fullfile(input_folder,'*.avi'));
    xlslist = dir(fullfile(input_folder,'*.xlsx'));

    txtdata = xlsread(strcat(input_folder,'\',xlslist(i).name)); %xlsread('times.xlsx');
    timerows = 1:3:size(txtdata,1);
    timestamps = txtdata(timerows,2);
    led_warmup_time = timestamps(1)/1000; %time stamp of led warm up in sec
    stim_times = timestamps(2:2:length(timestamps))/1000; %time stamps for stim in sec
    exp_end_times = timestamps(3:2:length(timestamps))/1000; %time stamp for trial end in sec
    
    vidname = strsplit(vidlist(1).name,'.avi');
    vidname = vidname{1};
    
    rawvideo=VideoReader(strcat(input_folder,'\',vidlist(i).name));
    time_increment = 1/rawvideo.FrameRate; %video step in time in seconds
    
    wt = waitbar(0,'starting video analysis2');%progress bar to see how code processsteps=len;
    steps = rawvideo.FrameRate*rawvideo.Duration;%total frames
    frame = 1;
    tolerance = 0.025; %0.01 sec tolerance to segment videos
    vidcount = 1; %how many videos are written
    
    %%
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
    
    %%
    while hasFrame(rawvideo)
       single_frame = readFrame(rawvideo); %grab current frame 
       
       if abs(led_warmup_time - rawvideo.CurrentTime) <= tolerance
           fprintf('Reached LED warmup time\n')
           led_warmup_time
           v = VideoWriter(strcat(input_folder,'\',exptype,'\',vidname,'_',num2str(vidcount),' segment.avi'));
           v.FrameRate = rawvideo.FrameRate;
           open(v)
           vidcount = vidcount + 1;
       end

       if vidcount > 1
           writeVideo(v,single_frame)
           
           if vidcount - 1 < length(exp_end_times)
               if frame == exp_end_frame(vidcount-1)%abs(exp_end_times(vidcount - 1) - rawvideo.CurrentTime) <= tolerance
                  fprintf('End of video segmentation for trial %1.0f\n',vidcount - 1)
                  close(v)

                  v = VideoWriter(strcat(input_folder,'\',exptype,'\',vidname,'_',num2str(vidcount),' segment.avi'));
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
    output_folder = strcat(input_folder,exptype);
end