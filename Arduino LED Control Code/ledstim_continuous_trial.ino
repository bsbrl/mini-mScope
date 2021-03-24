const byte triggerin = 7; //trigger from controller to DAQ to start videos
const byte triggerout = 8; //ttl pulse from DAQ to controller when a frame is taken
unsigned long duration;

float globaltime;
float globaltime2;
float starttime;
float starttime2;
float previousMillis = 0; // will store last time LED was updated
float previousMillis2 = 0; // store last time buzzstim was updated
//const double interval = 16.667; //camera framerate interval// interval at which to blink (milliseconds)
const double blueled_time = 20; //led blinking interval in ms
const double greenled_time = 4; // green led blinking interval
const double buzzstim_time = 5; // buzzer stim interval in ms; value in Hz is 1/buzzstim_time*1000

int count = 1;
int count2 = 1;
int totalloop = 1; // how many times to run through each stimulus pair
int incomingByte = 0;   // for incoming serial data
int testvar = 0;

const byte blueled = 3; // blue led at pin 9 
const byte greenled = 10; // green led at pin 10
const byte buzzstim = 6; // vibration stimulus at pin 6
const byte ledpin = 9; //led to segment videos on macroscope
const byte ledstim = 4; //stimulus led
const byte macroscope_trig = 2; //trig for external macroscope
//int totaltrial_repeat = 1; //total trials to repeat each stimulus pair under total trials (i.e. repeat visual and sound 12 times for 3 sets of data collection)
int totaltrials = 50; //total trials

float ledtime; //time to turn on led before next trial starts (sec)
//int trialtime = 15; //trial time in seconds
//int trialcount = 1; //counter for trial iterations
//int pausetime = 20000; //total video duration

int ledState = HIGH;
int ledState2 = HIGH;
int pulsecount = 1;
int greencount = 1;
int bluecount = 1;
int buzzcount = 1;

float exptime = 8; //data collection segment in seconds
float pulse_time = 3; //when to start each pulse in seconds
float pulse_duration = 0.1; //pulse duration for stimulus in seconds
float pausetime = 0; //pause time between trials in seconds
float led_warmup_time = 120; //experimentally determined led warm up period in seconds
float var;

float exptime_change = exptime;
float pulse_timechange = pulse_time;
//int trigdelay = 1;

int usercount = 1;
int val = 0;
int first_trigval = 0;
int blue_ledval = 0;
int green_ledval = 0;
int j;
int triggerwrite = 1;
int statementcount = 1;
int led_warmup_count = 1;
float temptime;


unsigned long temp;
unsigned long temp2;

volatile byte state = LOW;

unsigned int finals[3];

void setup() {
  // put your setup code here, to run once:
  pinMode(triggerin,OUTPUT);
  pinMode(triggerout,INPUT);
  pinMode(blueled,OUTPUT);
  pinMode(greenled,OUTPUT);
  pinMode(buzzstim, OUTPUT);
  pinMode(ledpin,OUTPUT);
  pinMode(ledstim,OUTPUT);
  pinMode(macroscope_trig,OUTPUT);
  attachInterrupt(digitalPinToInterrupt(triggerout),PulseLed,CHANGE);
  Serial.begin(19200);
  //Serial.println("Enter any key to begin the program");
}

void loop(){
  if (testvar == 0){
    incomingByte = Serial.read();
    usercount += 1;
    if (usercount % 100000 == 0){
      Serial.println("Pending user input... enter y to begin ");
      //Serial.println(interval*1000);
    }
    
    if (incomingByte != -1){ //if user input is non zero
      testvar = 1;
      //Serial.println(usercount);
    }
  }
  

  if (testvar == 1){
    if (totalloop <= totaltrials){
      starttime2 = micros();
      if (triggerwrite == 1){
        digitalWrite(triggerin,HIGH); // trigger DAQ to start recording
        //digitalWrite(ledpin,LOW);
        //digitalWrite(macroscope_trig,HIGH);
        
        if (totalloop==1){
          digitalWrite(ledpin,HIGH);
        }
        
        triggerwrite = 2;
      }
  
      if (count == 1){ // take first instance after trigger sent and record this ttl value
        first_trigval = state;
        count = 2;
      }
  
      if (count == 2){
          if (state != first_trigval){ //take first instance where the ttl pulse value changes 
            Serial.println(" ");
            Serial.println("Starting video now: ");
            Serial.println(" ");
            
            globaltime = micros(); // take teensy clock 
            starttime = globaltime;
//            Serial.print("Global time time is ");
//            temptime = globaltime / 1000000;
//            Serial.print(temptime);
//            Serial.println(" seconds");
//            Serial.println(" ");
            globaltime2 = globaltime;
            count = 3;

            blue_ledval = state; //assign first different value to be blue led (assume first bright frame captured to be blue led)
            green_ledval = first_trigval; //assign 0 state ttl pulse to be green led
            if (totalloop == 1){
              digitalWrite(ledpin,LOW);
            }
          }
        }
        if (totalloop == 1){
            pulse_timechange = pulse_time + led_warmup_time;
            exptime_change = exptime + led_warmup_time;
          }
          
        if (count == 3){
          if (starttime <= (exptime_change + globaltime/1000000) * 1000000){ //run led pulsing for experiment duration
            if (led_warmup_count == 1){
              if (starttime >= (led_warmup_time + globaltime/1000000) * 1000000){
                //digitalWrite(macroscope_trig,HIGH);
                Serial.print("LEDs warmed up");
                Serial.println(" ");
                Serial.print("Current time is ");
                if (totalloop == 1){
                  temptime = (starttime - globaltime) / 1000;
                }else{
                  temptime = (starttime - globaltime2) / 1000;
                }
                
                Serial.print(temptime);
                Serial.println(" milliseconds");
                Serial.println(" ");
                led_warmup_count = 2;
                
              }
            }
            float currentMillis = micros();
            starttime = currentMillis;
            
    
            if (((pulse_timechange + globaltime/1000000)*1000000)<starttime && (starttime<=(pulse_timechange + pulse_duration + globaltime/1000000)*1000000)){
              digitalWrite(ledstim,HIGH);
              if (statementcount == 1){
                Serial.print("LED Stimulus ");
                Serial.println(totalloop);
                Serial.print("Current time is ");
                if (totalloop == 1){
                  temptime = (starttime - globaltime) / 1000;
                }else{
                  temptime = (starttime - globaltime2) / 1000;
                }
//                finals[totalloop] = round(temptime*100);
                Serial.print(temptime);
                Serial.println(" milliseconds");
                Serial.println(" ");
                statementcount = 2;
                
              }
            }else{
              digitalWrite(ledstim,LOW);
            }
              
            if (blue_ledval == state){ //if blue led ttl is equal to current read ttl value
              if (bluecount == 1){ // if "on" counter for blue led is true
                previousMillis = currentMillis;
                digitalWrite(blueled,HIGH); //turn blue led on
                //digitalWrite(macroscope_trig,HIGH);
                //ledState = LOW;//switch led state
                bluecount = 2; //change counter to "off" for blueu led
              }
              if (currentMillis - previousMillis >= blueled_time*1000){ //if time grows beyond desired for blue led pulse
                digitalWrite(blueled,LOW); //turn led off
                //digitalWrite(macroscope_trig,LOW);
                greencount = 1; //assume a switch to green led now
                //ledState2 = HIGH;
              }
            }else{ //run when ttl value changes (i.e. switch from green to blue)
              
              if (greencount == 1){
                previousMillis = currentMillis; //repeat above with green led
                digitalWrite(greenled,HIGH);
                //digitalWrite(macroscope_trig,HIGH);
                //ledState2 = LOW;
                greencount = 2;
                
              }
              if (currentMillis - previousMillis >= greenled_time*1000){
                digitalWrite(greenled,LOW);
                //digitalWrite(macroscope_trig,LOW);
                bluecount = 1;
                //ledState = HIGH;
              }
            }
          }else{
//            for (j=1;j<=1000;j++){
//              digitalWrite(triggerin,LOW);
//            }

            //digitalWrite(triggerin,LOW);
            if (statementcount == 2){
              Serial.print("End of Trial ");
              Serial.println(totalloop);
              Serial.print("Current time is ");
              if (totalloop == 1){
                  temptime = (starttime - globaltime) / 1000;
                }else{
                  temptime = (starttime - globaltime2) / 1000;
                }
              Serial.print(temptime);
              Serial.println(" milliseconds");
              Serial.println(" ");
              
              //digitalWrite(greenled,LOW);
              //digitalWrite(blueled,LOW);
              statementcount = 1;
            }
            
            
            if (starttime <= (exptime_change + pausetime + globaltime/1000000)*1000000){
              starttime = micros();
            }else{
              if (count2 == 1){
                starttime2 = micros();
                count2 = 2;
                ledtime = starttime2 + 10000;
              }
              if ( starttime <= ledtime ){
                starttime = micros();
                //digitalWrite(ledpin,HIGH);
              }else{
                //digitalWrite(ledpin,LOW);
                starttime = micros();
                exptime_change = starttime/1000000 + exptime;
                pulse_timechange = starttime/1000000 + pulse_time;
                //globaltime = 0;
                
                
                totalloop += 1;
                triggerwrite = 1;
                count2 = 1;
              }
//              starttime = micros();
//              exptime_change = starttime/1000000 + exptime;
//              pulse_timechange = starttime/1000000 + pulse_time;
              globaltime = 0;
              starttime = micros();
//              
//              
//              totalloop += 1;
//              triggerwrite = 1;
            }
            
          }
        }
      }else{ //if outside experiment time
        if (totalloop == totaltrials+1){
          Serial.println("Experiment has concluded");
          digitalWrite(blueled,LOW); //shut down leds and conclude experiment
          digitalWrite(greenled,LOW);
          digitalWrite(triggerin,LOW);
          digitalWrite(ledpin,LOW);
//          for (int i = 0; i <= totaltrials;i++){
//            Serial.print(finals[i]);
//            Serial.print(" ");
//          }
          //delay(1000);
          totalloop += 1;
          testvar = 2;
        }
    }
  }
}



void PulseLed(){
  state = digitalRead(triggerout);
  //delayMicroseconds(2000);
}
