= Dataset <Teori>
I dette prosjektet bruker vi DroneRF-datasettet, som består av radiofrekvensopptak fra kommersielle droner under kontrollerte forhold. Datasettet ble publisert sammen med artikkelen til Al-Sa'd et al., der formålet var å bygge en åpen database for RF-basert dronedeteksjon og droneidentifikasjon. Datasettet inneholder opptak fra tre dronetyper: Parrot Bebop, Parrot AR Drone og DJI Phantom. I tillegg inneholder datasettet bakgrunnsopptak uten aktiv drone.

Hvert opptak er merket med en BUI-kode, som er en bitstreng som beskriver både dronetypen og tilstanden til dronen. De første bitene identifiserer dronen, mens de siste bitene beskriver dronens modus. I vårt prosjekt brukes disse kodene til å lage tre mulige klassifikasjonsmål: `target_binary`, `target_family` og `target_mode`. `target_binary` skiller mellom bakgrunn og drone, `target_family` skiller mellom dronefamilier, og `target_mode` beskriver en mer detaljert 10-klasse klassifisering av dronemodus. Hovedfokuset i prosjektet er `target_mode`.

Dronene i datasettet kommuniserer i 2.4 GHz-båndet. Siden én mottaker ikke kunne dekke hele signalbåndet samtidig, ble opptakene gjort med to RF-mottakere. Den ene mottakeren registrerte den lave delen av frekvensbåndet, merket `L`, mens den andre registrerte den høye delen, merket `H`. Datasettene inneholder derfor ofte to tilhørende filer for samme opptak: én for lavt bånd og én for høyt bånd.

Hver fil inneholder rå tidsdomenedata lagret som CSV-verdier. En typisk fil kan for eksempel hete `10100L_3.csv`. Denne filen kan tolkes som et opptak fra dronen identifisert av `101`, i modus `00`, fra lavfrekvensbåndet `L`, og som segment nummer 3 i opptakssekvensen. På denne måten inneholder filnavnet både label-informasjon og informasjon om hvilken del av RF-opptaket filen tilhører.

== Datakvalitet og utfordringer

DroneRF-datasettet har flere egenskaper som gjør det utfordrende å bruke direkte i maskinlæring. For det første består opptakene av svært lange tidsserier. En enkelt CSV-fil inneholder mange millioner målepunkter, og det er derfor ikke praktisk å bruke hele filer direkte som individuelle treningspunkter. Signalene må deles opp i mindre vinduer før de kan mates inn i modellene.

For det andre er datasettet høy-dimensjonalt. Selv korte vinduer av rå RF-data kan inneholde tusenvis av verdier. Dette gir mye informasjon, men øker også risikoen for overtilpasning og gjør modelltreningen mer krevende. Derfor må vi velge en representasjon som balanserer informasjonsinnhold og beregningskostnad.

En tredje utfordring er klasseubalanse. Noen droner og moduser forekommer flere ganger enn andre. Dette kan gjøre at en modell lærer å prioritere majoritetsklassene, samtidig som den gjør det dårligere på klasser med færre eksempler. Av den grunn evaluerer vi ikke modellene kun med accuracy, men også med macro-F1 og confusion matrix.

En fjerde utfordring er at enkelte klasser kan være overlappende i RF-mønster. To ulike moduser kan gi ganske like radiosignaturer, særlig dersom de tilhører samme dronefamilie eller bruker samme kommunikasjonsprotokoll. Dette gjør modusklassifisering vanskeligere enn ren drone/ikke-drone-deteksjon. Den opprinnelige DroneRF-studien viser også at klassifikasjonsytelsen synker når antall klasser øker, noe som tyder på at de mer detaljerte klassene er vanskeligere å skille.



== Utvelging av features fra datasettet
  Hvert opptak er knyttet til en BUI-kode der de tre første bitene identifiserer
  maskinvaren og de to siste beskriver dronens modus:

  - Modus 1 (00): Påslått og tilkoblet kontroller.
  - Modus 2 (01): Automatisk sveveflyging.
  - Modus 3 (10): Flyging uten videoopptak.
  - Modus 4 (11): Flyging med videoopptak.

#image("../images/bui.png")
  
Dronene kommuniserer over WiFi#sym.trademark på 2,4 GHz, med en total båndbredde på opptil 80 MHz ifølge @DroneRF_dataset. Siden én mottaker kun hadde 40 MHz øyeblikkelig båndbredde, ble to mottakere benyttet parallelt — én for lavt bånd (L) og én for høyt bånd (H). Med 40 MHz båndbredde registrerer mottakeren 40 000 000 målinger per sekund. Siden hver fil inneholder 10 000 000 verdier, representerer én fil 0,25 sekunder med opptak.

I @time_11000L_0 og @time_11000H_0 vises råsignalet i tidsdomenet for BUI 11000, for henholdsvis lavt og høyt bånd. Signalene inneholder kun informasjonssignalet rundt bærefrekvensen, selve bærefrekvensen er ikke synlig i tidsdomenet. Vi ser også at det lave båndet har høyere amplitude i dette opptaket.

#figure(
  image("../images/11000L_0.png"),
  caption: [Plott av 11000L_0.csv i tidsdomenet]
)<time_11000L_0>
#figure(
image("../images/11000H_0.png"),
caption: [Plott av 11000H_0.csv i tidsdomenet]
)<time_11000H_0>

Det er ikke nødvendig å rekonstruere ett komplett RF-signal fra de to båndene for maskinlæringsformål. Modellen kan fint finne struktur i de separate signalene, så lenge alle opptak er gjort med identisk utstyr og format, noe som er tilfellet i dette datasettet.

En mer robust tilnærming enn tidsdomenet er å se på frekvenssignaturen til hver dronetype og modus. Tidsdomene-signalet forventes å variere betydelig fra opptak til opptak ettersom informasjonen sendes til ulike tidspunkt, mens frekvensinnholdet er mer stabilt siden dronene kommuniserer på produsentspesifiserte frekvenser. Dronekommunikasjon er en form for frekvensmodulasjon, noe som betyr at informasjonen er kodet i frekvensene.

For å hente ut frekvensinnholdet bruker vi Rask Fourier-transformasjon (FFT). Det er imidlertid ikke mulig å rekonstruere det faktiske RF-frekvensspekteret fra disse opptakene. CSV-filene inneholder kun den reelle delen av signalet, ikke den imaginære. Uten en komplett In-fase/Kvadratur (IQ)-representasjon kan vi ikke skille mellom positive og negative frekvenser, de negative frekvensene brettes derfor over til den positive siden i FFT-resultatet. Det vi får er et basebånd-frekvensspekter fra 0 til 20 MHz, der vi ikke kan vite hvilke frekvenser som er brettet og hvilke ikke. Sammenligningen mellom tidsdomenet og frekvensdomenet for filen `11000L_0.csv` er vist i @freq_1000L_0.

#figure(
  image("../img/11000L_0_freq.png", width: 90%),
  caption: [Plott av tids- og frekvensdomene for 11000L_0]
)<freq_1000L_0>

Dette betyr at vi ikke kan lese av et fysisk meningsfylt RF-spekter, men all frekvensinformasjon er likevel bevart i en konsistent form. For modellen er dette tilstrekkelig: den trenger ikke forstå den fysiske betydningen av frekvensene, kun at alle segmenter presenteres på nøyaktig samme måte under trening og testing — noe som er garantert siden alle opptak bruker identisk utstyr og format.

For å utnytte informasjonen i frekvensspekteret best mulig, behandles inndataene ulikt avhengig av hvilken arkitektur som skal utføre klassifiseringen. Begge modellene benytter den samme grunnleggende signalrepresentasjonen: glidende vinduisering av råsignalet med en fast vindusstørrelse på 64 000 sampler og 50 % overlapping (stride = 32 000 sampler). Med 40 MHz samplingsfrekvens tilsvarer hvert vindu 1,6 millisekunder. For hvert vindu beregnes gjennomsnittlig magnitudeenergi per frekvensbånd ved hjelp av FFT, der 32 frekvensbånd fordeles jevnt over det positive frekvensspekteret fra 0 til 20 MHz. L- og H-båndet behandles som separate kanaler og gir til sammen 64 frekvensbånds-energier per vindu. Avhengig av opptakets lengde genereres mellom noen titalls og over 300 vinduer per fil; alle opptak paddes eller avkortes til nøyaktig 300 vinduer for å gi en uniform representasjon.

For CNN-modellen presenteres sekvensen av 300 vinduer direkte som en todimensjonal tensoreinngang av form (300, 64). Modellen kan dermed selv oppdage temporale mønstre — endringer i frekvensprofilen over tid — uten at det kreves manuell feature-konstruksjon.

For MLP-modellen reduseres de 300 vinduene til én flat feature-vektor ved hjelp av statistisk pooling. For hvert av de 64 frekvensbåndene beregnes tre statistikker på tvers av vinduene: gjennomsnitt, standardavvik og maksimum. Dette gir til sammen 192 egenskaper per opptak (64 bånd × 3 statistikker), som vist i @mlp_features_table. Gjennomsnittsverdien representerer den typiske frekvensprofilen for opptaket, standardavviket fanger temporal variasjon, og maksimum bevarer informasjon om toppenergi som et gjennomsnitt ville skjult.

#figure(
  caption: [Oversikt over de 192 egenskapene (features) for MLP-modellen, beregnet ved statistisk pooling av 300 frekvensbånds-energivinduer.],
  table(
    columns: (1fr, 1fr, 2fr),
    inset: 7pt,
    align: (left, left, left),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { gray.lighten(40%) },

    [*Kategori*], [*Feature-navn*], [*Beskrivelse*],

    table.cell(rowspan: 3, align: horizon)[*Lavt bånd (L), 32 bånd*],
    [mean\_L\_band00 -- mean\_L\_band31], [Gjennomsnittlig energi per frekvensbånd over 300 vinduer.],
    [std\_L\_band00 -- std\_L\_band31],  [Standardavvik per frekvensbånd — fanger temporal variasjon.],
    [max\_L\_band00 -- max\_L\_band31],  [Maksimal energi per frekvensbånd — bevarer toppaktivitet.],

    table.cell(rowspan: 3, align: horizon)[*Høyt bånd (H), 32 bånd*],
    [mean\_H\_band00 -- mean\_H\_band31], [Gjennomsnittlig energi per frekvensbånd over 300 vinduer.],
    [std\_H\_band00 -- std\_H\_band31],  [Standardavvik per frekvensbånd — fanger temporal variasjon.],
    [max\_H\_band00 -- max\_H\_band31],  [Maksimal energi per frekvensbånd — bevarer toppaktivitet.],
  )
) <mlp_features_table>


== Eksplorativ Dataanalyse (EDA)

Datasettet består av 227 opptak fordelt over fem klasser, som vist i @class_distribution_table. Klasse 0 er den største med 63 opptak, mens klasse 4 er den minste med 39. Denne ubalansen er moderat — den største klassen er om lag 1,6 ganger større enn den minste — men tilstrekkelig til at evaluering utelukkende basert på nøyaktighet kan gi et misvisende bilde av modellens ytelse. Av den grunn benyttes macro-F1 som primær evalueringsmetrikk, vektet slik at alle klasser teller likt.

#figure(
  caption: [Klassefordeling i DroneRF-datasettet (N = 227).],
  table(
    columns: (auto, auto, auto, auto),
    inset: 7pt,
    align: (center, left, center, center),
    stroke: 0.5pt + gray,
    fill: (x, y) => if y == 0 { gray.lighten(40%) },
    [*Klasse*], [*Dronemodus (BUI-kode)*], [*Antall opptak*], [*Andel*],
    [0], [Påslått og tilkoblet (00)], [63], [27,8 %],
    [1], [Automatisk svev (01)], [41], [18,1 %],
    [2], [Flyging uten video (10)], [42], [18,5 %],
    [3], [Flyging med video (11)], [42], [18,5 %],
    [4], [Bakgrunn (ingen drone)], [39], [17,2 %],
  )
) <class_distribution_table>

Et viktig teknisk poeng ved frekvensbånds-representasjonen er bruken av reell FFT (rFFT). Siden CSV-filene kun inneholder den reelle delen av signalet — uten imaginær IQ-komponent — brettes de negative frekvensene over til den positive siden i FFT-resultatet. Det resulterende basebånd-frekvensspekteret strekker seg fra 0 til 20 MHz, og de 32 frekvensbåndene fordeles jevnt over dette området. Selv om vi dermed ikke kan rekonstruere det fysiske RF-spekteret, er frekvensinformasjonen bevart i en konsistent form på tvers av alle opptak, noe som er tilstrekkelig for klassifisering.

Frekvensbånds-energiene har svært lave absoluttverdier, typisk i størrelsesorden $10^(-4)$ til $10^(-3)$, og energifordelingen varierer betydelig mellom opptakene. Enkelte opptak viser kraftige toppverdier som avviker markant fra medianen, særlig i lavfrekvensbåndet (L). Denne utliggerprofilen gjør RobustScaler — som normaliserer basert på median og interkvartilspredning — til et mer egnet valg enn StandardScaler for MLP-modellen, siden StandardScaler er sensitiv overfor slike energitopper.


== Train/test-splitt
Datasettet ble delt i trenings- og testdata før modelltrening. Vi brukte en stratified train/test-splitt, slik at klassefordelingen i størst mulig grad ble bevart i både trenings- og testsettet. Dette er spesielt viktig ved modusklassifisering, siden klassene kan være ubalanserte.

En viktig utfordring med DroneRF-datasettet er risiko for data leakage. Siden lange opptak deles inn i mange mindre vinduer, kan svært like segmenter ende opp i både trenings- og testsett dersom splitten gjøres etter vinduing. Dette kan gi kunstig høy ytelse, fordi modellen da testes på signaler som ligner sterkt på signaler den allerede har sett under trening. For å redusere denne risikoen bør splitten ideelt sett gjøres på opptaks- eller filnivå før vinduene genereres. Dersom dette ikke er fullstendig implementert, må det nevnes som en begrensning i prosjektet.

I vår implementasjon bruker vi en fast random seed for å gjøre splitten reproduserbar. Teststørrelsen er satt til 20 %, mens resten brukes til trening. For modellene brukes testsettet også som valideringsgrunnlag under trening, blant annet for å lagre beste modell basert på valideringsytelse. Dette diskuteres videre som en metodisk begrensning, siden et separat valideringssett ville gitt en renere evaluering.


== Forbehandling

Før modelltrening ble råfilene lest inn og koblet til korrekte klasser basert på BUI-koden i filnavnet. For hvert opptak ble L- og H-signalene prosessert med den samme glidende vinduesfunksjonen: vindusstørrelse 64 000 sampler, 50 % overlapping. Fra hvert vindu beregnes 64 frekvensbånds-energier (32 per kanal) via rFFT med Hanning-vindusvekting for å redusere spektral lekkasje. Alle opptak paddes eller avkortes til nøyaktig 300 vinduer.

For CNN-modellen presenteres de 300 vinduene direkte som en (300, 64)-tensor. Normalisering utføres per opptak ved å dele på den største absoluttverdien i tensoren, noe som sikrer at inngangsverdiene er i området [−1, 1] uten å endre relative forskjeller mellom frekvensbåndene.

For MLP-modellen aggregeres de 300 vinduene til én flat vektor med 192 egenskaper ved beregning av gjennomsnitt, standardavvik og maksimum per frekvensbånd. Denne vektoren normaliseres med RobustScaler, tilpasset eksklusivt til treningssettet, for å unngå datalekkasje fra testsettet.

Klasseubalansen håndteres ved frekvensbasert prøvevekting: for MLP sendes beregnede sample-vekter (compute_sample_weight) til MLPClassifier under trening, og for CNN benyttes tilsvarende klassevekter (class_weight) i Keras. Begge metodene øker den effektive innflytelsen til de mindre representerte klassene under gradientoppdateringene uten å endre datasettets sammensetning.
