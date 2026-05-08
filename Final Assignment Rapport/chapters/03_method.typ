= Metode <Design>
Først gjennomføres en eksplorativ dataanalyse for å identifisere nødvendig datarensing og preprosessering før trening. Deretter deles dataene opp i et 80/20 trenings-/testsett. Som baseline bygges en "kitchen sink"-modell der alle tilgjengelige variabler brukes uten preprosessering, for å kunne sammenligne og vurdere ytelsen til den endelige modellen.
To typer maskinlæringsalgoritmer vil bli brukt: MLP og CNN. MLP fordi denne fungerer godt på strukturerte/tabulære data og kan lære komplekse, ikke-lineære sammenhenger mellom input-funksjoner. CNN fordi denne kan finne lokale mønstre og romlige avhengigheter i data (for eksempel i bilder eller sekvenser), noe som gir bedre gjenkjenning av mønstre.

Purpose: Explain how you approached the problem and why you made each choice.

== Forberedelse av data
=== MLP
Dataene ble normalisert ved hjelp av StandardScaler for å sikre at RF-signaler med ulike måleskalaer og potensielle utliggere fikk en felles distribusjon, noe som er kritisk for at MLP-algoritmen skal konvergere effektivt. Det ble ikke benyttet windowing eller resizing av bilder da datasettet bestod av numeriske egenskaper ekstrahert direkte fra signalene. For å håndtere det begrensede datasettet på ca. 227 rader, ble klassebalansering ivaretatt gjennom stratifisert splitting (stratify=y), slik at hver drone-type ble representert proporsjonalt i både trenings- og testsettet.

=== CNN
Råsignalene ble ikke feature-ekstrahert manuelt, men bevart direkte som tidsseriedata. Hvert opptak ble delt inn i vinduer av fast lengde ved hjelp av en glidende vinduing med 50 % overlapping (stride = window_size / 2), slik at hvert vindu ble ett individuelt treningspunkt. Det lave og det høye frekvensbåndet ble representert som to separate inndatakanaler, organisert som ein tensor med form (2, sekvenslengde). Normalisering ble utført per kanal basert på gjennomsnitt og standardavvik beregnet eksklusivt fra treningssettet, for å forhindre datalekkasje fra testsettet. Klassebalansen ble håndtert gjennom frekvensbasert klassevekting under trening, fremfor stratifisert splitting alene.

== Funksjonsoppbygning
=== MLP
Feature-konstruksjonen tok utgangspunkt i domenespesifikke RF-trekk som frekvenstopper og signalstyrke. For å redusere modellens kompleksitet og fjerne støy, ble funksjonen filter_features benyttet for å ekskludere variabler med lav varians, noe som fungerte som en målrettet dimensjonalitetsreduksjon. Denne manuelle filtreringen sikret at modellen fokuserte på de mest distinkte kjennetegnene ved hver drone fremfor tilfeldige svingninger i bakgrunnsstøyen.

=== CNN
CNN-modellen opererer direkte på rå signalverdier uten manuell feature-konstruksjon. Gjennom vinduingen genereres betydelig flere treningspunkter enn antall originalfiler tilsier, ettersom hvert opptak bidrar med flere overlappende vinduer. Dette er avgjørende siden datasettet kun inneholder 227 originalopptak. Den todimensjonale inndatastrukturen, med H- og L-båndet som separate kanaler langs første akse og tidspunkter langs andre akse, lar modellen selv oppdage relevante mønstre i begge frekvensbåndene og på tvers av dem, uten at det er nødvendig å forhåndsdefinere hvilke frekvenser som er diskriminerende.

== Maskinlærings-algoritme og valg av modell
=== MLP
MLP (Multi-Layer Perceptron) ble valgt fordi den er i stand til å modellere komplekse, ikke-lineære sammenhenger som ofte finnes i RF-signaturer. Modellen antar at det finnes gjenkjennbare mønstre i de normaliserte dataene, og ble foretrukket over enklere lineære modeller for å kunne fange opp nyanserte forskjeller mellom dronene. Som vist i bilde.png, avslørte modellen utfordringer med å skille visse overlappende klasser (som Drone 0 og 2), noe som bekreftet at en enkel "Kitchen Sink"-tilnærming var utilstrekkelig og at arkitekturen krevde nøye finjustering.

=== CNN
CNN (Convolutional Neural Network) ble valgt fordi den er spesialisert på å detektere lokale mønstre i sekvensielle data, noe som er velegnet for RF-signaler der karakteristiske strukturer kan opptre på ulike tidspunkter i opptaket. Arkitekturen som ble benyttet er DroneCNNMultiTask, en endimensjonal faltningsmodell med flere konvolusjonsblokker etterfulgt av adaptiv gjennomsnittspooling og to separate klassifikasjonshoveder. Modellen er utformet som en multitask-modell som simultant predikerer både dronefamilie og dronemodus. Begrunnelsen for dette er at felles representasjonslæring på tvers av de to målvariablene kan gi en mer robust intern representasjon av signalet, ettersom dronefamilie og dronemodus er tett knyttet til hverandre i RF-signaturen. Dette gir modellen mulighet til å utnytte informasjon som er nyttig for begge oppgavene, og kan potensielt bedre generaliseringsevnen sammenlignet med en modell som kun predikerer modus alene.

== Hyperparameter-strategi
=== MLP
Det ble benyttet en systematisk GridSearchCV-strategi for å utforske kombinasjoner av arkitektur, regularisering og læringsrater. Denne strategien ble supplert med manuell finjustering basert på valideringskurver; når gapet mellom trenings- og testnøyaktighet ble for stort (overfitting), ble søkerommet justert mot enklere modeller med færre nevroner og høyere straffeparametere. Målet var å finne et "sweet spot" der modellen hverken memorerte støyen eller ble for enkel til å lære signalene.

=== CNN
Hyperparameterne ble justert manuelt med utgangspunkt i valideringskurver, fremfor en automatisert søkestrategi. De viktigste parameterne var antall og størrelse på konvolusjonskanaler (standard: 32, 64, 128), kjernestørrelse (7), droppout-rate (0,30), læringsrate (0,001) og vektforfall (1 × 10⁻⁴) som implisitt L2-regularisering. Vindusstørrelse og stride ble holdt faste for å sikre reproduserbarhet. Tapet fra de to klassifikasjonshovedene ble vektet likt (1,0 for henholdsvis familie- og modushodet), da det ikke var grunnlag for å prioritere det ene fremfor det andre uten ytterligere eksperimentering. Justeringsprosessen tok sikte på å redusere gapet mellom trenings- og valideringsytelse, der høy treningsnøyaktighet kombinert med lav validerings-F1 ble tolket som et signal om å redusere modellkapasiteten eller øke regulariseringen.

== Trenings-prosedyre
=== MLP
Treningen benyttet en log-loss kostfunksjon med adam-optimalisator og en adaptiv læringsrate for å sikre stabil og nøyaktig konvergens. For å motvirke overfitting ble L2-regularisering (Alpha) kombinert med early stopping, som avbrøt treningen dersom valideringsresultatene ikke forbedret seg over 75-150 iterasjoner. Dette sikret at modellen ble frosset ved sin beste generaliseringsevne fremfor å fortsette treningen til 100 % nøyaktighet på treningsdataene.

=== CNN
Treningen benyttet kryssentropi-tap for begge klassifikasjonshoveder, kombinert med Adam-optimalisatoren og vektforfall som implisitt L2-regularisering for å motvirke overtilpasning. For å håndtere klasseubalansen ble det benyttet frekvensbaserte klassevekter, beregnet som invertert frekvens, slik at sjeldne moduser fikk forholdsmessig høyere innflytelse på gradienten. Treningen ble avbrutt ved tidlig stopp basert på kombinert macro-F1 på valideringssettet, med en tålmodighetsparameter på 10 epoker uten forbedring. Modellvektene fra den best observerte epoken ble restaurert ved treningsslutt, slik at den endelige modellen representerte det punktet med best generaliseringsevne og ikke nødvendigvis siste iterasjon.