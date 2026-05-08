= Metode <Design>
Dataene deles 80/20 i trenings- og testsett med stratifisert splitting, slik at klassefordelingen bevares i begge settene. Som baseline brukes en Dummy Classifier (mest-hyppig-strategi) og en uregulert "kitchen sink"-modell, for å etablere yttergrenser og vise at rå kapasitet ikke er tilstrekkelig. To modeller sammenlignes: MLP på aggregerte features og CNN på vinduesekvenser.

== Forberedelse av data
=== MLP
De 300 vinduene per opptak komprimeres til én flat feature-vektor med 192 egenskaper ved å beregne tre statistikker per frekvensbånd: gjennomsnitt, standardavvik og maksimum. Gjennomsnittsverdien representerer den typiske frekvensprofilen for opptaket. Standardavviket fanger om aktiviteten er stabil eller skiftende fra vindu til vindu. Maksimum bevarer informasjon om toppenergi som et gjennomsnitt ville skjult. Vektoren normaliseres med RobustScaler tilpasset eksklusivt til treningssettet, for å unngå datalekkasje. Klassebalansen ivaretas med stratifisert splitting og frekvensbasert prøvevekting (compute_sample_weight) under trening.

=== CNN
Hvert opptak representeres direkte som en (300, 64)-matrise, der modellen selv kan oppdage mønstre i hvordan frekvensprofilen endrer seg gjennom opptaket. Normalisering gjøres per opptak ved å dele på den største absoluttverdien i matrisen, noe som bringer alle inngangsverdier til området [−1, 1] uten å endre de relative forholdene mellom frekvensbåndene. Klasseubalansen håndteres med frekvensbasert klassevekting (class_weight) beregnet fra treningssettet.

== Maskinlærings-algoritme og valg av modell
=== MLP
MLP ble valgt fordi den er i stand til å modellere ikke-lineære sammenhenger mellom de aggregerte frekvensbånds-egenskapene, og fordi den gir en naturlig sammenligningspartner til CNN: begge opererer på den samme underliggende vinduerepresentasjonen av signalet, men der CNN behandler sekvensen direkte, samler MLP informasjonen via statistisk pooling. Denne forskjellen gjør det mulig å isolere bidraget fra sekvensinformasjonen.

=== CNN
CNN ble valgt fordi den er spesialisert på å finne lokale mønstre i sekvensielle data, noe som passer for RF-signaler der karakteristiske strukturer kan opptre på ulike tidspunkter i opptaket. Arkitekturen er en endimensjonal faltningsmodell med to konvolusjonsblokker (16 og 32 filtre med kjernestørrelser 7 og 5), global gjennomsnittspooling, ett fullt tilkoblet lag med 32 nevroner og fem utgangsnoder. BatchNormalization etter hvert konvolusjonsblokk stabiliserer treningen. Modellen er bevisst holdt liten — om lag 11 000 parametere — basert på tommelfingerregelen om at antall parametere bør stå i rimelig forhold til antall treningspunkter.

== Hyperparameter-strategi
=== MLP
GridSearchCV med 5-fold stratifisert kryssvalidering og macro-F1 som scoringsmetrikk utforsket 32 kombinasjoner av arkitektur ((32,), (64,), (32, 16), (64, 32)), regularisering (α ∈ {0,05; 0,1; 0,5; 1,0}) og læringsrate (0,001 og 0,0005). Macro-F1 ble valgt som scoringsmetrikk i stedet for nøyaktighet fordi nøyaktighet er misvisende ved klasseubalanse — en modell som alltid predikerer majoritetsklassen oppnår over 27 % uten å lære noe. Med 181 treningsfiler favoriserte søket konsekvent de minste og sterkest regulariserte modellene.

=== CNN
Hyperparameterne ble justert manuelt med utgangspunkt i valideringskurver. Høy treningsnøyaktighet kombinert med lav valideringsnøyaktighet ble tolket som signal om for høy modellkapasitet eller for svak regularisering. Dropout-rater 0,4 etter hvert konvolusjonsblokk og 0,5 etter det tett tilkoblede laget, L2-straff α = 0,005 på alle vekter, og et GaussianNoise-lag (σ = 0,05) som innebygd augmentering viste seg å gi et stabilt gap mellom trenings- og valideringskurver. Læringsrate 0,0005 og batchstørrelse 16 ble beholdt gjennom alle forsøk.

== Trenings-prosedyre
=== MLP
Log-loss kostfunksjon med Adam-optimalisatoren og adaptiv læringsrate. Tidlig stopp med tålmodighetsparameter 50 iterasjoner uten forbedring sørget for at modellen ble fryst ved beste generaliseringsevne. Frekvensbaserte prøvevekter (compute_sample_weight) ble sendt gjennom sklearn Pipeline til MLPClassifier, slik at sjeldne klasser fikk forholdsmessig høyere innflytelse på gradientoppdateringene. Et internt valideringssett på 15 % av treningsdataene overvåket early stopping.

=== CNN
Kryssentropi-tap med Adam-optimalisatoren og L2-straff på alle vekter. Frekvensbaserte klassevekter (class_weight) økte den effektive innflytelsen til underrepresenterte moduser. Tidlig stopp med tålmodighetsparameter 20 epoker overvåket valideringstapet, og beste modellvekter ble restaurert ved treningsslutt. Maks 200 epoker.
