//-------------------------------User Defined Variables-------------------------------------------------------------------------------------------
const double blueled_time = 20; //led blinking interval in ms
const double greenled_time = 20; // green led blinking interval in ms
int totaltrials = 1; //total trials to run for specified experiment time
float exptime = 200; //data collection segment in seconds
float led_warmup_time = 120; //experimentally determined led warm up period in seconds

//notes
//1. LED pulse times have to be at least ~5ms less than the framerate set on the Miniscope camera (CMOS actual framerate may vary by +- ~5ms from set framerate)
//2. LED warmup time will add to the total experiment time for the first trial if there are more than 1 total trials set. This means if the experiment
//time is set 10 seconds per trial for 3 total trials, LED warmup time will run 120sec first before the first trial of 10 seconds. Then each subsequent
//trial will only last for the desired 10 seconds.
//3. For long experiment collection sections, you can simply change the total trials to 1 trial and set the experiment time to something less than 200 seconds
//as the total experiment time (LED warmup + exp_time*totaltrials) should be less than 350-400 seconds total to prevent LED  burnout.
//4. Note that at the end of each trial, the software will continue to record video so the LEDs maintain a constant temperature. The serial monitor
//will print the time at the end of each trial and these times can be passed to a script we can provide to auto-segment the video
//into individual trials if more than one trial duration has been set above



//-------------------------------Main Variables-------------------------------------------------------------------------------------------------------

const byte triggerin = 7; //trigger from controller to DAQ to start videos
const byte triggerout = 8; //ttl pulse from DAQ to controller when a frame is taken
const byte macroscope_trig = 2; //trig for external macroscope
unsigned long duration;

float globaltime;
float globaltime2;
float starttime;
float starttime2;
float previousMillis = 0; // will store last time LED was updated
float previousMillis2 = 0; // store last time buzzstim was updated


int count = 1;
int count2 = 1;
int totalloop = 1; // how many times to run through each stimulus pair
int incomingByte = 0;   // for incoming serial data

int testvar = 0;
const byte blueled = 3; // blue led at pin 3
const byte greenled = 10; // green led at pin 10


float ledtime; //time to turn on led before next trial starts (sec)

int ledState = HIGH;
int ledState2 = HIGH;
int pulsecount = 1;
int greencount = 1;
int bluecount = 1;
int buzzcount = 1;


float pulse_time = 5; //when to start pulse in seconds
float pulse_duration = 1; //pulse duration for stimulus in seconds
float pausetime = 0; //pause time between trials in seconds
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

void setup() {
  // put your setup code here, to run once:
  pinMode(triggerin, OUTPUT);
  pinMode(triggerout, INPUT);
  pinMode(blueled, OUTPUT);
  pinMode(greenled, OUTPUT);
  pinMode(macroscope_trig, OUTPUT);
  attachInterrupt(digitalPinToInterrupt(triggerout), PulseLed, CHANGE);
  Serial.begin(19200);

  //Serial.println("Enter any key to begin the program");
}

void loop() {
  if (testvar == 0) {
    incomingByte = Serial.read();
    usercount += 1;
    if (usercount % 100000 == 0) {
      Serial.println("Pending user input... enter y to begin ");
      //Serial.println(interval*1000);
    }

    if (incomingByte != -1) { //if user input is non zero
      testvar = 1;
      //Serial.println(usercount);
    }
  }


  if (testvar == 1) {
    if (totalloop <= totaltrials) {
      starttime2 = micros();
      if (triggerwrite == 1) {
        digitalWrite(triggerin, HIGH); // trigger DAQ to start recording

        triggerwrite = 2;
      }

      if (count == 1) { // take first instance after trigger sent and record this ttl value
        first_trigval = state;
        count = 2;
      }

      if (count == 2) {
        if (state != first_trigval) { //take first instance where the ttl pulse value changes
          Serial.println(" ");
          Serial.println("Starting video now: ");
          Serial.println(" ");

          globaltime = micros(); // take teensy clock
          starttime = globaltime;
          globaltime2 = globaltime;
          count = 3;
          //Serial.print("current val ");
          //Serial.print(val);
          //Serial.println(" ");
          blue_ledval = state; //assign first different value to be blue led (assume first bright frame captured to be blue led)
          green_ledval = first_trigval; //assign 0 state ttl pulse to be green led

          //Serial.println(blue_ledval);
          //Serial.println(green_ledval);
        }
      }
      if (totalloop == 1) {
        pulse_timechange = pulse_time + led_warmup_time;
        exptime_change = exptime + led_warmup_time;
      }

      if (count == 3) {
        if (starttime <= (exptime_change + globaltime / 1000000) * 1000000) { //run led pulsing for experiment duration
          if (led_warmup_count == 1) {
            if (starttime >= (led_warmup_time + globaltime / 1000000) * 1000000) {
              //digitalWrite(macroscope_trig,HIGH);
              Serial.print("LEDs warmed up");
              Serial.println(" ");
              Serial.print("Current time is ");
              if (totalloop == 1) {
                temptime = (starttime - globaltime) / 1000;
              } else {
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
          statementcount = 1;
          if (blue_ledval == state) { //if blue led ttl is equal to current read ttl value

            if (bluecount == 1) { // if "on" counter for blue led is true
              previousMillis = currentMillis;
              digitalWrite(blueled, HIGH); //turn blue led on
              digitalWrite(macroscope_trig, HIGH);
              //ledState = LOW;//switch led state
              bluecount = 2; //change counter to "off" for blueu led
            }
            if (currentMillis - previousMillis >= blueled_time * 1000) { //if time grows beyond desired for blue led pulse
              digitalWrite(blueled, LOW); //turn led off
              digitalWrite(macroscope_trig, LOW);
              greencount = 1; //assume a switch to green led now
              //ledState2 = HIGH;
            }
          } else { //run when ttl value changes (i.e. switch from green to blue)

            if (greencount == 1) {

              previousMillis = currentMillis; //repeat above with green led
              digitalWrite(greenled, HIGH);
              digitalWrite(macroscope_trig, HIGH);
              //ledState2 = LOW;
              greencount = 2;

            }
            if (currentMillis - previousMillis >= blueled_time * 1000) {
              digitalWrite(greenled, LOW);
              digitalWrite(macroscope_trig, LOW);
              bluecount = 1;
              //ledState = HIGH;
            }
          }
        } else {
          //            for (j=1;j<=1000;j++){
          //              digitalWrite(triggerin,LOW);
          //            }

          //digitalWrite(triggerin,LOW);
          if (statementcount == 1) {
            Serial.print("End of Trial ");
            Serial.println(totalloop);
            Serial.print("Current time is ");
            if (totalloop == 1) {
              temptime = (starttime - globaltime) / 1000;
            } else {
              temptime = (starttime - globaltime2) / 1000;
            }
            Serial.print(temptime);
            Serial.println(" milliseconds");
            Serial.println(" ");
            //digitalWrite(greenled,LOW);
            //digitalWrite(blueled,LOW);
            statementcount = 2;
          }


          if (starttime <= (exptime_change + pausetime + globaltime / 1000000) * 1000000) {
            starttime = micros();
          } else {
            if (count2 == 1) {
              starttime2 = micros();
              count2 = 2;
              ledtime = starttime2 + 0;//10000;
            }
            if ( starttime <= ledtime ) {
              starttime = micros();
            } else {
              //digitalWrite(ledpin,LOW);
              starttime = micros();
              exptime_change = starttime / 1000000 + exptime;
              pulse_timechange = starttime / 1000000 + pulse_time;
              globaltime = 0;


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
    } else { //if outside experiment time
      if (totalloop == totaltrials + 1) {
        Serial.println("Experiment has concluded");
        digitalWrite(blueled, LOW); //shut down leds and conclude experiment
        digitalWrite(greenled, LOW);
        digitalWrite(triggerin, LOW);
        digitalWrite(macroscope_trig, LOW);
        totalloop += 1;
        testvar = 2;
      }
    }
  }
}



void PulseLed() {
  state = digitalRead(triggerout);
  digitalWrite(macroscope_trig, LOW);
  //delayMicroseconds(2000);
}
