= Metode <Design>
Først gjennomføres en eksplorativ dataanalyse for å identifisere nødvendig datarensing og preprosessering før trening. Deretter deles dataene opp i et 80/20 trenings-/testsett. Som baseline bygges en "kitchen sink"-modell der alle tilgjengelige variabler brukes uten preprosessering, for å kunne sammenligne og vurdere ytelsen til den endelige modellen.
To typer maskinlæringsalgoritmer vil bli brukt: MLP og CNN. MLP fordi denne fungerer godt på strukturerte/tabulære data og kan lære komplekse, ikke-lineære sammenhenger mellom input-funksjoner. CNN fordi denne kan finne lokale mønstre og romlige avhengigheter i data (for eksempel i bilder eller sekvenser), noe som gir bedre gjenkjenning av mønstre.


== Forberedelse av data
=== MLP
Dataene ble normalisert ved hjelp av RobustScaler, som benytter median og interkvartilspredning i stedet for gjennomsnitt og standardavvik. Dette gjør normaliseringen mer robust overfor energitopper i RF-signaler, som kan gi kraftige utliggere dersom StandardScaler benyttes. Råsignalene ble prosessert med den samme glidende vinduesfunksjonen som CNN-modellen, og de 300 vinduene per opptak ble deretter komprimert til én flat feature-vektor ved hjelp av tre statistiske mål per frekvensbånd: gjennomsnitt, standardavvik og maksimum. Dette gir til sammen 192 egenskaper per opptak (32 bånd × 2 kanaler × 3 statistikker). Gjennomsnittsverdien representerer den typiske frekvensprofilen, standardavviket fanger variasjon, og maksimum bevarer toppenergi uavhengig av tidspunkt. Klassebalansering ble ivaretatt gjennom stratifisert splitting (stratify=y) kombinert med frekvensbasert prøvevekting (sample_weight) under trening.

=== CNN
Råsignalene ble prosessert med den samme glidende vinduesfunksjonen som MLP, og hvert opptak ble representert som en matrise av form (300, 64) — 300 vinduer med 64 frekvensbånds-energier per vindu. Normalisering ble utført per opptak ved å dele på den største absoluttverdien i matrisen, slik at inngangsverdiene havner i området [−1, 1] uten å endre de relative forskjellene mellom frekvensbåndene. Klassebalansen ble håndtert gjennom frekvensbasert klassevekting (class_weight) under trening.

== Funksjonsoppbygning
=== MLP
Feature-konstruksjonen benytter den samme glidende vinduingen som CNN-modellen. De 300 vinduene per opptak representerer RF-signalet som en tidsoppløst sekvens av frekvensbåndsenergier. Fra disse vinduene beregnes tre statistiske sammendrag per frekvensbånd — gjennomsnitt, standardavvik og maksimum — noe som gir 192 egenskaper per opptak.

=== CNN
CNN-modellen mottar de 300 frekvensbånds-energivinduene direkte som en sekvens og kan finne mønstre i hvordan frekvensprofilen endrer seg gjennom opptaket, uten manuell feature-konstruksjon. Vinduingen gir også flere treningspunkter per originalfil, noe som er nyttig siden datasettet kun inneholder 227 opptak.

== Maskinlærings-algoritme og valg av modell
=== MLP
MLP (Multi-Layer Perceptron) ble valgt fordi den er i stand til å modellere komplekse, ikke-lineære sammenhenger mellom de statistiske egenskapene ekstrahert fra frekvensbåndsvinduene. Modellen ble foretrukket over enklere lineære modeller for å fange opp nyanserte forskjeller mellom dronemodusen, og gir en naturlig sammenligning med CNN-modellen ettersom begge opererer på den samme underliggende vinduerepresentasjonen av signalet — men der CNN behandler vindusekvensen direkte, samler MLP informasjonen via statistisk pooling.

=== CNN
CNN ble valgt fordi den er god på å finne lokale mønstre i sekvensielle data, noe som passer for RF-signaler der karakteristiske strukturer kan opptre på ulike tidspunkter i opptaket. Arkitekturen er en endimensjonal faltningsmodell med to konvolusjonsblokker (16 og 32 filtre) etterfulgt av global gjennomsnittspooling og ett fullt tilkoblet lag med 32 nevroner. Den endelige utgangen er ett klassifikasjonslagmed fem noder (én per dronemodus). Modellen er bevisst holdt liten — om lag 11 000 parametere — for å passe til et treningssett på 181 opptak.

== Hyperparameter-strategi
=== MLP
Det ble benyttet en systematisk GridSearchCV-strategi med 5-fold stratifisert kryssvalidering for å utforske kombinasjoner av arkitektur, regularisering og læringsrater. Søkerommet dekket fire arkitekturalternativer — (32,), (64,), (32, 16) og (64, 32) — kombinert med fire regulariseringsverdier (α ∈ {0,05; 0,1; 0,5; 1,0}) og to læringsrater (0,001 og 0,0005). Scoringsmetrikken ble satt til macro-F1 i stedet for nøyaktighet, da ubalanserte klasser gjør at nøyaktighet er et misvisende mål. Med kun 181 treningsfiler favoriserte søket konsekvent de minste og sterkest regulariserte modellene, noe som bekreftet at datasettets størrelse setter et tak for nyttig modellkapasitet.

=== CNN
Hyperparameterne ble justert manuelt med utgangspunkt i valideringskurver. Antall filtre (16 og 32), kjernestørrelser (7 og 5), dropout-rater (0,4 og 0,5) og L2-straff (α = 0,005) ble valgt for å holde modellen enkel nok til å generalisere fra 181 treningsfiler. Et GaussianNoise-lag med σ = 0,05 ble lagt til som innebygd augmentering for å gjøre modellen mer robust. Læringsraten ble satt til 0,0005 og batchstørrelsen til 16. Utgangspunktet for justeringen var at høy treningsnøyaktighet kombinert med lav valideringsnøyaktighet indikerte behov for sterkere regularisering eller lavere modellkapasitet.

== Trenings-prosedyre
=== MLP
Treningen benyttet log-loss kostfunksjon med Adam-optimalisatoren og adaptiv læringsrate for stabil konvergens. L2-regularisering (α) ble satt av GridSearchCV, og tidlig stopp med en tålmodighetsparameter på 50 iterasjoner uten forbedring sørget for at modellen ble fryst ved sin beste generaliseringsevne. Klasseubalansen ble kompensert ved å beregne frekvensbaserte prøvevekter (compute_sample_weight) på treningssettet og sende disse gjennom Pipeline-objektet til MLPClassifier under trening. Modellen ble trent med et internt valideringssett på 15 % av treningsdataene for early stopping-overvåkingen.

=== CNN
Treningen benyttet kryssentropi-tap med Adam-optimalisatoren og L2-straff på alle vekter. Frekvensbaserte klassevekter sørget for at sjeldne moduser fikk større innflytelse på gradientoppdateringene. Tidlig stopp med tålmodighetsparameter 20 overvåket valideringstapet, og beste modellvekter ble restaurert ved treningsslutt. Maksimalt antall epoker ble satt til 200.