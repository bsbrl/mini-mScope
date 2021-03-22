clear;clc;close all;
folder = 'E:\abf files\m3\';
atf_files = dir(fullfile(folder,'*.atf'));

for i = 1:length(atf_files)
    name = atf_files(i).name;
    fprintf('Analyzing file with name below\n')
    name
    data = table2array(readtable(strcat(folder,name),'FileType', 'Text', 'Delimiter', '\t', 'HeaderLines', 10));
    time = data(:,1);
    stimpulse = data(:,3);
    onpulse = data(:,2);
    
    [~,pulse_high] = findpeaks(stimpulse,'MinPeakHeight',4);
    [~,time_zero] = findpeaks(onpulse,'MinPeakHeight',3.2); %on voltage from microcontroller
    time_end = time(time_zero(end));
    time_zero = time(time_zero(1)); %only store first point
    time_end = time_end - time_zero; %rescale time to 0
    
    plot(time-time_zero,stimpulse)
    hold on
    plot(time(pulse_high)-time_zero,stimpulse(pulse_high),'*r')
    pulse_high_split = diff(pulse_high);
    first_pulse = find(pulse_high_split > 100)+1;
    first_pulse = [1;first_pulse];
    plot(time(pulse_high(first_pulse))-time_zero,stimpulse(pulse_high(first_pulse)),'*g')
    hold on
    plot(time-time_zero,onpulse);
    pulse_start = time(pulse_high(first_pulse));
    
    exp_start_time = time(pulse_high(first_pulse)-5000);
    exp_end_time = time(pulse_high(first_pulse)+5000);
    time_stamp_diff = time_end - (exp_end_time(end)-time_zero); %find gap in time at end of exp
    
    data_start = stimpulse(pulse_high(first_pulse)-5000);
    data_end = stimpulse(pulse_high(first_pulse)+5000); %shift 5 sec
    plot(exp_start_time-time_zero,data_start,'*k')
    plot(exp_end_time-time_zero,data_end,'*m')
    
    name = strsplit(name,'.atf');
    name = name{1};
    xlswrite(strcat(folder,name,'.xlsx'),(exp_start_time(1)-time_zero)*1000,1,'D4');
    row_count = 7;
    for j = 1:length(exp_start_time)
        xlswrite(strcat(folder,name,'.xlsx'),(pulse_start(j)-time_zero)*1000,1,strcat('D',num2str(row_count)));
        row_count = row_count + 3; %next write
        xlswrite(strcat(folder,name,'.xlsx'),(exp_end_time(j)-time_zero)*1000,1,strcat('D',num2str(row_count)));
        row_count = row_count + 3; %next write
    end
    pause(2)
    close;
   
    
end