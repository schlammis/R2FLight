
Channel 1 of the signal generator is connected to the resistor.
Channel 2 to the capacitor
Voltage 1 is the reistance voltage

ACQ3:VOLT 0.3,(@101:104)
SAMP3:RATE 800000,(@101:104)
SAMP3:COUN 80000,(@101:104)
INP3:COUP AC,(@101:104)
TRIG3:SLOP POS, (@101:104)
TRIG3:SOUR EXT,(@101)
TRIG3:SOUR (@101),(@102:104)
INIT3 (@101:104)



-----------------
The following code is from Fetch_binary.vi
-----------------

FORM3 REAL
ACQ3:VOLT 0.3,(@101:104)
SAMP3:RATE 800000,(@101:104)
SAMP3:COUN 800000,(@101:104)
INP3:COUP AC,(@101:104)
TRIG3:SLOP POS, (@101:104)
TRIG3:SOUR EXT,(@101)
TRIG3:SOUR (@101),(@102:104)
INIT3 (@101:104)


FETCH3? (@102)


