= Metode <Design>
For at klassefordelingen skal bevares i begge settene, deles dataene 80/20 med stratifisert splitting. Som baseline brukes en Dummy Classifier (mest-hyppig-strategi). To modeller sammenlignes: MLP på sammensatte frekvens-statistikker og CNN på vindusekvenser.

== Signalrepresentasjon og vindusvalg
Hvert opptak segmenteres i glidende vinduer på 64 000 sampler med 50 % overlapping, noe som gir om lag 300 vinduer per fil. Vinduestørrelsen på 64 000 sampler er et kompromiss: kortere vinduer gir finere tidsoppløsning men svakere frekvensoppløsning (og mer støy i FFT-estimatene), mens lengre vinduer gir det motsatte. Med 40 MHz samplingsfrekvens svarer 64 000 sampler til 1,6 ms — tilstrekkelig til å fange én full RF-ramme fra dronen. 50 % overlapping gir nok vinduer til at statistiske estimater over sekvensen blir stabile.

Hvert vindu omregnes til 32 frekvensbånds-energier per kanal (L og H) via rFFT med Hanning-vekting, noe som gir 64 energier per vindu. Antall bånd (32) er valgt fordi det er grovt nok til å redusere dimensjonaliteten fra 32 768 FFT-koeffisienter, men fint nok til å beholde informasjon om ulike frekvenssegmenter i båndet. Alle opptak paddes eller avkortes til 300 vinduer for å sikre lik inputstørrelse.

== Forberedelse av data
=== MLP
De 300 vinduene per opptak komprimeres til én flat feature-vektor med 192 egenskaper ved å beregne tre statistikker per frekvensbånd: gjennomsnitt, standardavvik og maksimum. Gjennomsnittsverdien representerer den typiske frekvensprofilen for opptaket. Standardavviket fanger om aktiviteten er stabil eller skiftende fra vindu til vindu. Maksimum bevarer informasjon om toppenergi som et gjennomsnitt ville skjult. Vektoren normaliseres med RobustScaler tilpasset eksklusivt til treningssettet for å unngå datalekkasje, altså at informasjon fra testdataene 'smitter' over i treningsprosessen og gir et urealistisk bilde av hvor god modellen faktisk er. For at modellen skal bli like god på alle de forskjellige dronene, er det implementert to tiltak:

  - Vi har sørget for at både treningssettet og testsettet inneholder den samme blandingen av alle dronetypene.

  - Vi har gitt de sjeldne dronene ekstra betydning under treningen, slik at modellen må ta dem på alvor selv om det finnes færre eksempler av dem.


=== CNN
Hvert opptak representeres direkte som en (300, 64)-matrise, der modellen selv kan oppdage mønstre i hvordan frekvensprofilen endrer seg gjennom opptaket. Normalisering gjøres per opptak ved å dele på den største absoluttverdien i matrisen, noe som bringer alle inngangsverdier til området [−1, 1] uten å endre de relative forholdene mellom frekvensbåndene. Klasseubalansen håndteres med frekvensbasert klassevekting (class_weight) beregnet fra treningssettet.


== Maskinlærings-algoritme og valg av modell
=== MLP
Med 192 sammensatte egenskaper er inputformatet tabelldata. Naturlige alternativer er Random Forest og SVM, som begge fungerer godt på tabelldata med mange egenskaper og lite data. Vi valgte MLP fremfor disse fordi MLP modellerer ikke-lineære sammenhenger mellom egenskapene på en måte som er mer direkte sammenlignbar med CNN — begge er nevrale nett som optimeres med gradientnedstigning. Random Forest og SVM ville vært interessante referansepunkter, men falt utenfor prosjektets omfang.

=== CNN
Et naturlig alternativ for sekvensielle data er Long Short-Term Memory (LSTM)  eller andre rekurrente nettverk (RNN). Disse er spesialdesignet for å fange opp langtidsavhengigheter, altså mønstre som utvikler seg over lengre tid i en sekvens. Imidlertid krever slike modeller ofte store datamengder for å trene stabilt. For dette prosjektet antar vi at de mest relevante kjennetegnene ligger i de lokale endringene i frekvensenergi fra vindu til vindu, heller enn i globale strukturer over lang tid. En 1D-CNN er derfor bedre egnet, da den effektivt søker etter slike lokale mønstre ved bruk av faste filtre. CNN er dessuten langt mer dataeffektiv; vi har bevisst holdt modellen liten (ca. 11 000 parametere) for å sikre at den lærer de generelle kjennetegnene i signalene fremfor å memorere spesifikke eksempler fra treningssettet.

== Hyperparameter-strategi
=== MLP
GridSearchCV (en automatisert prosess som systematisk tester en rekke kombinasjoner av hyperparametere for å finne den konfigurasjonen som gir best resultat) med 5-fold stratifisert kryssvalidering og macro-F1 som scoringsmetrikk utforsket 32 kombinasjoner av arkitektur ((32,), (64,), (32, 16), (64, 32)), regularisering (α ∈ {0,05; 0,1; 0,5; 1,0}) og læringsrate (0,001 og 0,0005). Macro-F1 ble valgt som scoringsmetrikk i stedet for nøyaktighet fordi nøyaktighet er misvisende ved klasseubalanse — en modell som alltid predikerer majoritetsklassen oppnår over 27 % uten å lære noe. Med 181 treningsfiler favoriserte søket konsekvent de minste og sterkest regulariserte modellene.


=== CNN
Modellens hyperparametere ble justert manuelt ved å observere trenings- og valideringskurvene underveis. Dersom modellen viste høy nøyaktighet på treningsdata, men lav nøyaktighet på valideringsdata, ble dette tolket som overtilpasning (overfitting). For å motvirke dette ble det utforsket ulike kombinasjoner av filtre (fra 8/16 til 32/64), dropout-rater (0,2–0,5) og L2-regularisering (0,001–0,01). Læringsraten (0,0005) og batchstørrelsen (16) ble fastsatt tidlig i prosessen for å sikre stabil læring. Den optimale konfigurasjonen ble funnet med 16 og 32 filtre i de to konvolusjonslagene, kombinert med flere regulariseringstiltak. Det ble benyttet en dropout-rate på 0,4 etter hver konvolusjonsblokk og 0,5 før det tett tilkoblede laget. I tillegg ble det lagt til en L2-straff ($alpha = 0,005$) på alle vekter; en metode som hindrer at enkelte vekter i modellen blir unødig store, noe som tvinger modellen til å spre læringen over flere parametere for å generalisere bedre. Sammen med et GaussianNoise-lag ($sigma = 0,05$), som skaper kunstig variasjon i signalene (dataaugmentering), ga disse tiltakene et stabilt og lite gap mellom trenings- og valideringskurvene.

== Trenings-prosedyre
=== MLP
Modellen ble trent med funksjonen _Log-loss_ og optimaliseringsalgoritmen Adam, som benytter en adaptiv læringsrate for mer effektiv oppdatering av vektene. For å sikre best mulig generaliseringsevne ble det brukt «Early Stopping», en funksjon som overvåket et eget kontrollsett (15 % av dataene) og stanset treningen dersom modellen ikke viste forbedring i løpet av 50 iterasjoner. Siden datasettet har en ujevn fordeling av dronetyper, ble det brukt prøvevekter (compute_sample_weight) i en sklearn Pipeline. Dette sørget for at sjeldne klasser fikk større betydning under treningen, slik at modellen lærte å identifisere alle klassene like presist.
=== CNN
For CNN-modellen ble funksjonen _kryssentropi_ benyttet sammen med Adam-optimalisatoren. For å hindre overtilpasning ble det lagt til en L2-straff på alle vekter, som fungerer som en matematisk begrensning for å holde modellen mer robust. Fordi enkelte dronemoduser hadde færre eksempler enn andre, ble det brukt klassevekter (class_weight) for å sikre at disse fikk tilstrekkelig betydning under treningen. Prosessen ble overvåket med «Early Stopping» over maksimalt 200 epoker; dersom valideringstapet ikke ble forbedret i løpet av 20 epoker, ble treningen avbrutt og modellen tilbakestilt til de vektene som ga best resultat.
