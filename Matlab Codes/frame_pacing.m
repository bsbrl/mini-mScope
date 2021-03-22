function [final_frames,exp_time] = frame_pacing(input_folder,meso_camnum,beh_camnum,meso_framerate,beh_framerate)

files(1).name = 'timestamp.dat';

for i = 1:length(files)
    currentname = files(i).name;
    data = importdata(strcat(input_folder,currentname));
    
    exp_time = data.data(end,3)/1000; %final time stamp from miniscope in seconds
    frame_errorperc_meso = 10; %framerate +- 10%
    frame_errorperc_beh = 100+frame_errorperc_meso; %framerate err for beh cam (ignore single frame drops) on upperbound

    meso_totframes = exp_time * meso_framerate;
    beh_totframes = exp_time * beh_framerate;

    hist_scale = 0:1:100;
    upperbound_meso = ((frame_errorperc_meso/100)*(1/meso_framerate)+(1/meso_framerate))*1000;%#fps +-10% (cmos drops less frames)
    lowerbound_meso = ((1/meso_framerate)-(frame_errorperc_meso/100)*(1/meso_framerate))*1000;
    keep_framerange_meso = ceil([lowerbound_meso,upperbound_meso]);

    upperbound_beh = ((frame_errorperc_beh/100)*(1/beh_framerate)+(1/beh_framerate))*1000;%[fps-10%*FPS,fps+100*FPS] beh cam drops single frames very often, can ignore single frame drops since data for gcamp exists, beh doesnt occur at fast frequencies
    lowerbound_beh = ((1/beh_framerate)-(frame_errorperc_meso/100)*(1/beh_framerate))*1000;
    keep_framerange_beh = ceil([lowerbound_beh,upperbound_beh]);
    %data = importdata(strcat(folder,'timestamp.dat'));
    meso_ind = find(data.data(:,1) == meso_camnum); %find indices where cmos camera is stored 
    beh_ind = find(data.data(:,1) == beh_camnum);

    meso_frame = data.data(meso_ind,2);
    beh_frame = data.data(beh_ind,2);

    meso_time = data.data(meso_ind,3);
    beh_time = data.data(beh_ind,3);

    meso_framemax = max(meso_frame);
    beh_framemax = max(beh_frame);
    max_dropframe = min([meso_framemax,beh_framemax]);

    if max_dropframe == meso_framemax
        denom = meso_totframes;
    else
        denom = beh_totframes;
    end
    
    fprintf(strcat('Filename is' + " " + currentname,'\n'))

    fprintf('Total dropped frames in behavior cam is %4.0f/%4.0f\n',beh_totframes - beh_framemax,beh_totframes)
    fprintf('Total dropped frames in mesoscope cam is %4.0f/%4.0f\n',meso_totframes - meso_framemax,meso_totframes)
    fprintf('To be cautious, assuming maximum dropped frames is %1.0f\n\n',denom - max_dropframe)

    
    meso_timediff = diff(meso_time);
    beh_timediff = diff(beh_time);
    %[meso_timediff,meso_ind] = sort(diff(meso_time));%sort to elements by time difference stamps
    %[beh_timediff,beh_ind] = sort(diff(beh_time));

    figure(2*i-1)
    h = histogram(meso_timediff,hist_scale);
    xlim([20,60])
    xlabel('Frame Time (ms)')
    ylabel('Count')
    title('Mesoscope Camera Frame Pacing')
    
    figure(2*i)
    h2 = histogram(beh_timediff,hist_scale);
    xlim([20 60])
    xlabel('Frame Time (ms)')
    ylabel('Count')
    title('Behavior Camera Frame Pacing')
    
    keepmeso_ind = find(keep_framerange_meso(1)<=meso_timediff & meso_timediff<=keep_framerange_meso(2));
    keepbeh_ind = find(keep_framerange_beh(1)<=beh_timediff & beh_timediff<=keep_framerange_beh(2));

    final_frames_nonconsec = intersect(keepmeso_ind,keepbeh_ind);%index of frames that are shared between meso and beh cam with certainty
    
    %%
    final_frames = [];
    for k = 2:length(final_frames_nonconsec)-1
        if mod(final_frames_nonconsec(k),2) == 1 && final_frames_nonconsec(k) == final_frames_nonconsec(k-1)+1
           final_frames = [final_frames,final_frames_nonconsec(k),final_frames_nonconsec(k-1)];
            
        end

    end
    final_frames = sort(final_frames)';
    %%
    fprintf('Final common frames between cmos and behavior cameras are %1.0f/%1.0f\n',length(final_frames),meso_totframes)
end
