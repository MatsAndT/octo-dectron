= Metode <Design>
For at klassefordelingen skal bevares i begge settene, deles dataene 80/20 med stratifisert splitting. Som baseline brukes en Dummy Classifier (mest-hyppig-strategi) og en uregulert "kitchen sink"-modell. To modeller sammenlignes: MLP på sammensatte frekvens-statistikker og CNN på vindusekvenser.

== Forberedelse av data
=== MLP
De 300 vinduene per opptak komprimeres til én flat feature-vektor med 192 egenskaper ved å beregne tre statistikker per frekvensbånd: gjennomsnitt, standardavvik og maksimum. Gjennomsnittsverdien representerer den typiske frekvensprofilen for opptaket. Standardavviket fanger om aktiviteten er stabil eller skiftende fra vindu til vindu. Maksimum bevarer informasjon om toppenergi som et gjennomsnitt ville skjult. Vektoren normaliseres med RobustScaler tilpasset eksklusivt til treningssettet for å unngå datalekkasje, altså at informasjon fra testdataene 'smitter' over i treningsprosessen og gir et urealistisk bilde av hvor god modellen faktisk er. For at modellen skal bli like god på alle de forskjellige dronene, har vi gjort to ting:

  - Vi har passet på at både treningssettet og testsettet inneholder den samme blandingen av alle dronetypene.

  - Vi har gitt de sjeldne dronene ekstra betydning under treningen, slik at modellen må ta dem på alvor selv om det finnes færre eksempler av dem.

=== CNN
Hvert opptak representeres direkte som en (300, 64)-matrise, der modellen selv kan oppdage mønstre i hvordan frekvensprofilen endrer seg gjennom opptaket. Normalisering gjøres per opptak ved å dele på den største absoluttverdien i matrisen, noe som bringer alle inngangsverdier til området [−1, 1] uten å endre de relative forholdene mellom frekvensbåndene. Klasseubalansen håndteres med frekvensbasert klassevekting (class_weight) beregnet fra treningssettet.

== Maskinlærings-algoritme og valg av modell
=== MLP
MLP ble valgt fordi den er i stand til å modellere ikke-lineære sammenhenger mellom de sammensatte frekvensbånds-egenskapene, og fordi den gir en naturlig sammenligningspartner til CNN: begge opererer på den samme underliggende vinduerepresentasjonen av signalet, men der CNN behandler sekvensen direkte, samler MLP informasjonen via statistiske verdier for frekvensspekteret.

=== CNN
CNN ble valgt fordi den er svært god til å finne mønstre i signaler, uansett når i opptaket de dukker opp. Modellen bruker to lag med "filtre" for å fange opp detaljer, samt teknikker som gjør treningen mer stabil. Vi har bevisst holdt modellen liten (ca. 11 000 parametere) for at den skal lære de faktiske kjennetegnene til dronene fremfor å bare "pugge" treningsdataene.

== Hyperparameter-strategi
=== MLP
GridSearchCV med 5-fold stratifisert kryssvalidering og macro-F1 som scoringsmetrikk utforsket 32 kombinasjoner av arkitektur ((32,), (64,), (32, 16), (64, 32)), regularisering (α ∈ {0,05; 0,1; 0,5; 1,0}) og læringsrate (0,001 og 0,0005). Macro-F1 ble valgt som scoringsmetrikk i stedet for nøyaktighet fordi nøyaktighet er misvisende ved klasseubalanse — en modell som alltid predikerer majoritetsklassen oppnår over 27 % uten å lære noe basert på resultater fra Dummy Classifier. Med 181 treningsfiler i datasettet favoriserte søket konsekvent de minste og sterkest regulariserte modellene.

=== CNN
Vi justerte modellens hyperparametere for hånd ved å se på trenings- og valideringskurvene. Når treningsnøyaktigheten var høy, men valideringsnøyaktigheten lav, tolket vi det som at modellen overtilpasset (for høy kapasitet) eller at regulariseringen var for svak. For å redusere dette brukte vi:

    Dropout: 0,4 etter hver konvolusjonsblokk og 0,5 før det fullt tilkoblede laget.
    L2-regularisering: α = 0,005 på alle vekter.
    GaussianNoise (σ = 0,05) som innebygd dataaugmentering.

Disse tiltakene ga et mer stabilt og forutsigbart gap mellom trenings- og valideringskurvene. Læringsraten (0,0005) og batchstørrelsen (16) ble holdt faste i alle eksperimentene.

== Trenings-prosedyre
=== MLP
Log-loss kostfunksjon med Adam-optimalisatoren og adaptiv læringsrate. Toleranse for tidlig stopp på 50 iterasjoner uten forbedring sørget for at modellen ble stoppet ved beste generaliseringsevne og ikke ble overfitted. Frekvensbaserte prøvevekter ble sendt gjennom sklearn Pipeline til MLPClassifier, slik at sjeldne klasser fikk forholdsmessig høyere innflytelse på gradientoppdateringene. Et internt valideringssett på 15% av treningsdataene overvåket early stopping.

=== CNN
Kryssentropi-tap med Adam-optimalisatoren og L2-straff på alle vekter. Frekvensbaserte klassevekter  økte den effektive innflytelsen til underrepresenterte moduser. Toleranse for tidlig stopp på 20 epoker overvåket valideringstapet, og beste modellvekter ble restaurert ved treningsslutt. Maks 200 epoker.
