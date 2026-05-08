= Metode <Design>
Dataene deles 80/20 i trenings- og testsett. Som baseline brukes en "kitchen sink"-modell uten regularisering for å vise at rå kapasitet ikke er nok. To modeller sammenlignes: MLP på aggregerte features og CNN på vinduesekvenser.

== Forberedelse av data
=== MLP
De 300 vinduene per opptak komprimeres til 192 egenskaper ved tre statistikker per frekvensbånd (gjennomsnitt, standardavvik, maksimum). Normaliseringen gjøres med RobustScaler, som er robust overfor energitopper. Klassebalansen ivaretas med stratifisert splitting og frekvensbasert prøvevekting (sample_weight) under trening.

=== CNN
Hvert opptak representeres som en (300, 64)-matrise. Normalisering gjøres per opptak ved å dele på absolutt maksimum. Klasseubalansen håndteres med frekvensbasert klassevekting (class_weight).

== Maskinlærings-algoritme og valg av modell
=== MLP
MLP ble valgt for å modellere ikke-lineære sammenhenger mellom de aggregerte frekvensbånds-egenskapene. Der CNN behandler vinduesekvensen direkte, samler MLP informasjonen via statistisk pooling — dette gir en direkte sammenligning av de to representasjonsstrategiene.

=== CNN
CNN ble valgt fordi den er god på å finne lokale mønstre i sekvensielle data. Arkitekturen har to konvolusjonsblokker (16 og 32 filtre), global gjennomsnittspooling, ett fullt tilkoblet lag med 32 nevroner, og fem utgangsnoder. Modellen er bevisst holdt liten — om lag 11 000 parametere — for å passe til 181 treningsfiler.

== Hyperparameter-strategi
=== MLP
GridSearchCV med 5-fold stratifisert kryssvalidering og macro-F1 som scoringsmetrikk utforsket 32 kombinasjoner av arkitektur ((32,), (64,), (32,16), (64,32)), regularisering (α ∈ {0,05; 0,1; 0,5; 1,0}) og læringsrate (0,001 og 0,0005).

=== CNN
Hyperparameterne ble justert manuelt fra valideringskurver. Dropout-rater 0,4/0,5, L2-straff α = 0,005, GaussianNoise σ = 0,05, læringsrate 0,0005, batchstørrelse 16.

== Trenings-prosedyre
=== MLP
Log-loss med Adam og adaptiv læringsrate. Tidlig stopp med tålmodighet 50 iterasjoner. Internt valideringssett på 15 % av treningsdataene brukes til early stopping.

=== CNN
Kryssentropi-tap med Adam og L2-straff. Tidlig stopp med tålmodighet 20 epoker, beste vekter restaurert ved treningsslutt. Maks 200 epoker.
