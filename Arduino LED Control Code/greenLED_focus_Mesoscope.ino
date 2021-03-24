int i=0;
char buf[3];
int ledpin = 9;

void setup()
{
   Serial.begin(19200);
   pinMode(ledpin,OUTPUT);
}

void loop()
{
  if (Serial.available()>0){
   
   
    buf[i]= Serial.read();
   
    if (int(buf[i])==13 || int(buf[i])==11 ){  //If Carriage return has been reached
     
      int result=atoi(buf);
      Serial.print("PWM ");   
      Serial.println(result);     //print the converted char ar
      analogWrite(ledpin,result);
      if(result>255){
      Serial.println("Warning number too big");
      }
       
    for(int x=0;x<=3;x++){
    buf[x]=' ';
    }
    i=0;  //start over again
    } //if enter
     i++;
    } //IF Serial.available
 
 

}//LOOP
