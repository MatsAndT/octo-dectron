= Metode <Design>
For at klassefordelingen skal bevares i begge settene, deles dataene 80/20 med stratifisert splitting. Som baseline brukes en Dummy Classifier (mest-hyppig-strategi) og en uregulert "kitchen sink"-modell. To modeller sammenlignes: MLP på sammensatte frekvens-statistikker og CNN på vindusekvenser.

== Signalrepresentasjon og vindusvalg
Hvert opptak segmenteres i glidende vinduer på 64 000 sampler med 50 % overlapping, noe som gir om lag 300 vinduer per fil. Vinduestørrelsen på 64 000 sampler er et kompromiss: kortere vinduer gir finere tidsoppløsning men svakere frekvensoppløsning (og mer støy i FFT-estimatene), mens lengre vinduer gir det motsatte. Med 40 MHz samplingsfrekvens svarer 64 000 sampler til 1,6 ms — tilstrekkelig til å fange én full RF-ramme fra dronen. 50 % overlapping gir nok vinduer til at statistiske estimater over sekvensen blir stabile uten å øke datasettets minnefotavtrykk uforholdsmessig.

Hvert vindu omregnes til 32 frekvensbånds-energier per kanal (L og H) via rFFT med Hanning-vekting, noe som gir 64 energier per vindu. Antall bånd (32) er valgt fordi det er grovt nok til å redusere dimensjonaliteten fra 32 768 FFT-koeffisienter, men fint nok til å beholde informasjon om ulike frekvenssegmenter i båndet. Alle opptak paddes eller avkortes til 300 vinduer for å sikre lik inputstørrelse.

== Forberedelse av data
=== MLP
De 300 vinduene per opptak komprimeres til én flat feature-vektor med 192 egenskaper ved å beregne tre statistikker per frekvensbånd: gjennomsnitt, standardavvik og maksimum. Gjennomsnittsverdien representerer den typiske frekvensprofilen for opptaket. Standardavviket fanger om aktiviteten er stabil eller skiftende fra vindu til vindu. Maksimum bevarer informasjon om toppenergi som et gjennomsnitt ville skjult. Vektoren normaliseres med RobustScaler tilpasset eksklusivt til treningssettet for å unngå datalekkasje, altså at informasjon fra testdataene 'smitter' over i treningsprosessen og gir et urealistisk bilde av hvor god modellen faktisk er. For at modellen skal bli like god på alle de forskjellige dronene, har vi gjort to ting:

  - Vi har passet på at både treningssettet og testsettet inneholder den samme blandingen av alle dronetypene.

  - Vi har gitt de sjeldne dronene ekstra betydning under treningen, slik at modellen må ta dem på alvor selv om det finnes færre eksempler av dem.

=== CNN
Hvert opptak representeres direkte som en (300, 64)-matrise, der modellen selv kan oppdage mønstre i hvordan frekvensprofilen endrer seg gjennom opptaket. Normalisering gjøres per opptak ved å dele på den største absoluttverdien i matrisen, noe som bringer alle inngangsverdier til området [−1, 1] uten å endre de relative forholdene mellom frekvensbåndene. Klasseubalansen håndteres med frekvensbasert klassevekting (class_weight) beregnet fra treningssettet.

== Maskinlærings-algoritme og valg av modell
=== MLP
Med 192 sammensatte egenskaper er inputformatet tabelldata. Naturlige alternativer er Random Forest og SVM, som begge fungerer godt på tabelldata med mange egenskaper og lite data. Vi valgte MLP fremfor disse fordi MLP modellerer ikke-lineære sammenhenger mellom egenskapene på en måte som er mer direkte sammenlignbar med CNN — begge er nevrale nett som optimeres med gradientnedstigning. Random Forest og SVM ville vært interessante referansepunkter, men falt utenfor prosjektets omfang.

=== CNN
En naturlig alternativ modell for sekvensielle data er LSTM eller andre rekurrente nettverk (RNN). LSTM er designet for å modellere langtidsavhengigheter i sekvenser, men krever mer data for å trene stabilt. For dette datasettet er antagelsen at de lokale mønstrene i frekvensbånds-energiene — altså endringer fra vindu til vindu over korte spenn — er mer informative enn globale langtidsstrukturer. En 1D-CNN er bedre egnet for dette: den søker etter lokale mønstre med faste filterstørrelser og er langt mer dataeffektiv enn LSTM. Vi har bevisst holdt modellen liten (ca. 11 000 parametere) for at den skal generalisere fremfor å memorere treningsdataene.

== Hyperparameter-strategi
=== MLP
GridSearchCV med 5-fold stratifisert kryssvalidering og macro-F1 som scoringsmetrikk utforsket 32 kombinasjoner av arkitektur ((32,), (64,), (32, 16), (64, 32)), regularisering (α ∈ {0,05; 0,1; 0,5; 1,0}) og læringsrate (0,001 og 0,0005). Macro-F1 ble valgt som scoringsmetrikk i stedet for nøyaktighet fordi nøyaktighet er misvisende ved klasseubalanse — en modell som alltid predikerer majoritetsklassen oppnår over 27 % uten å lære noe basert på resultater fra Dummy Classifier. Med 181 treningsfiler i datasettet favoriserte søket konsekvent de minste og sterkest regulariserte modellene.

=== CNN
Vi justerte modellens hyperparametere manuelt ved å observere trenings- og valideringskurvene. Søkerommet som ble prøvd ut: antall filtre per lag (8/16, 16/32, 32/64), dropout-rate (0,2; 0,3; 0,4 etter konvolusjonsblokk, 0,3; 0,4; 0,5 før tett lag), og L2-straff (0,001; 0,005; 0,01). Læringsraten (0,0005) og batchstørrelse (16) ble satt etter de første forsøkene og holdt faste. Den beste konfigurasjonen ble:

  - Filtre: 16 og 32 i de to konvolusjonslagene
  - Dropout: 0,4 etter hver konvolusjonsblokk og 0,5 før tett lag
  - L2-regularisering: α = 0,005 på alle vekter
  - GaussianNoise (σ = 0,05) som innebygd dataaugmentering

Disse tiltakene ga et stabilt og lite gap mellom trenings- og valideringskurvene.

== Trenings-prosedyre
=== MLP
Log-loss kostfunksjon med Adam-optimalisatoren og adaptiv læringsrate. Toleranse for tidlig stopp på 50 iterasjoner uten forbedring sørget for at modellen ble stoppet ved beste generaliseringsevne og ikke ble overfitted. Frekvensbaserte prøvevekter ble sendt gjennom sklearn Pipeline til MLPClassifier, slik at sjeldne klasser fikk forholdsmessig høyere innflytelse på gradientoppdateringene. Et internt valideringssett på 15% av treningsdataene overvåket early stopping.

=== CNN
Kryssentropi-tap med Adam-optimalisatoren og L2-straff på alle vekter. Frekvensbaserte klassevekter  økte den effektive innflytelsen til underrepresenterte moduser. Toleranse for tidlig stopp på 20 epoker overvåket valideringstapet, og beste modellvekter ble restaurert ved treningsslutt. Maks 200 epoker.
