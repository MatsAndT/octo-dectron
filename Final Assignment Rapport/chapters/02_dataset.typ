= Datasett <Teori>
I dette prosjektet bruker vi DroneRF-datasettet, som består av RF-opptak fra kommersielle droner under kontrollerte forhold @DroneRF_dataset. Datasettet inneholder opptak fra tre dronetyper: Parrot Bebop, Parrot AR Drone og DJI Phantom, samt bakgrunnsopptak uten aktiv drone. Klasseubalansen er moderat — den største klassen (63 opptak) er om lag 1,6 ganger større enn den minste (39 opptak) — og evaluering gjøres derfor med macro-F1 i tillegg til nøyaktighet.

== Datastruktur og klassekoding
Hvert opptak er knyttet til en BUI-kode (Bit Unique Identifier) der de tre første bit'ene identifiserer maskinvaren og de to siste beskriver dronens modus (se @bui):

  - Modus 1 (00): Påslått og tilkoblet kontroller.
  - Modus 2 (01): Automatisk _hovering_.
  - Modus 3 (10): Flyging uten videoopptak.
  - Modus 4 (11): Flyging med videoopptak.

#figure(
  image("../images/bui.png"),
  caption: [Forklaring av BUI]
)<bui>

Dronene kommuniserer over WiFi på 2,4 GHz med en total båndbredde på opptil 80 MHz @DroneRF_dataset. Siden én mottaker i forskningsprosjektet som har laget datasettet hadde 40 MHz øyeblikkelig båndbredde ble to mottakere benyttet parallelt — én for lavt bånd (L) og én for høyt bånd (H). Med 40 MHz samplingsfrekvens og 10 000 000 verdier per fil representerer én fil i datasettet 0,25 sekunder.

@time_11000L_0 og @time_11000H_0 viser råsignalet for et eksempelopptak.

#figure(
  image("../images/11000L_0.png"),
  caption: [11000L_0.csv i tidsdomenet]
)<time_11000L_0>
#figure(
  image("../images/11000H_0.png"),
  caption: [11000H_0.csv i tidsdomenet]
)<time_11000H_0>

Frekvensinnholdet er mer stabilt enn tidsdomenet, siden dronene kommuniserer på faste frekvenser. Siden datasettet består av CSV-filer som kun inneholder de reelle amplitudeverdiene av signalet, og ikke komplekse IQ-data (in-phase/quadrature), oppstår det visse begrensninger i den spektrale analysen. Ved å benytte en reell fouriertransformasjon (rFFT) transformeres signalet til et basebånd-spekter i området 0 til 20 MHz per bånd (L eller H). Denne prosessen medfører to kritiske faktorer for tolkningen av dataene: 
- Tap av fortegn (Folding): Uten IQ-komponenter er det umulig å skille mellom positive og negative frekvensavvik relativt til bærebølgefrekvensen ($f_c$). Frekvenser som opprinnelig lå under bærebølgen, vil "foldes" over på den positive siden. Resultatet er et spekter som ikke gir et direkte bilde av de faktiske fysiske frekvensverdiene dronen sender på i RF-spekteret.
- Relative, kontra fysiske frekvenser: Spekteret vi observerer representerer derfor ikke de absolutte overføringsfrekvensene (i 2,4 GHz-båndet), men snarere en konsekvent, relativ representasjon av signalets båndbredde og moduleringsegenskaper slik de fremstår etter nedmiksing til basebånd.
For maskinlæring betyr dette at selv om spekteret ikke er "fysisk korrekt" i tradisjonell forstand, er den interne frekvensinformasjonen konsistent på tvers av alle opptak i datasettet. Maskinlæringsmodeller er svært effektive til å identifisere komplekse mønstre i slike relative data, så lenge de systematiske endringene (som folding) er like for alle klasser. Dette fordrer imidlertid at fremtidig bruk av modellen må benytte et identisk oppsett for opptak som det forskerne bak datasettet brukte. Modellen lærer ikke å kjenne dronen "i lufta", men snarere hvordan dronens signal ser ut gjennom akkurat denne spesifikke digitale "linsen" som er benyttet for å lage datasettet. @freq_1000L_0 viser sammenligningen mellom tid- og frekvensdomenet for en utvalgt CSV-fil fra datasettet.

#figure(
  image("../img/11000L_0_freq.png", width: 90%),
  caption: [Tids- og frekvensdomene for 11000L_0]
)<freq_1000L_0>


DroneRF-datasettet har noen egenskaper som gjør det utfordrende å bruke direkte i maskinlæring. Opptakene er svært detaljerte tidsserier — en fil inneholder 10 millioner målepunkter — og kan ikke brukes direkte som individuelle treningspunkter. Selv etter vindusfunksjonen er RF-dataen høydimensjonal, noe som øker risikoen for overfitting, særlig med så få treningseksempler. Klasseubalansen er moderat, men nok til at modeller kan lære å favorisere majoritetsklassene.

== Eksplorativ Dataanalyse (EDA)

Klassefordelingen er vist i @class_distribution_table. Frekvensbånds-energiene har svært lave absoluttverdier — typisk i størrelsesorden $10^(-4)$ til $10^(-3)$ — og enkeltopptak kan vise kraftige toppverdier som avviker markant fra medianen. Denne profilen, med et stabilt støygulv og sporadiske energitopper, er grunnen til at RobustScaler (basert på median og interkvartilspredning) er bedre egnet enn StandardScaler for MLP-normalisering.

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

== Forbehandling

Datasettet ble delt 80/20 i trenings- og testsett med stratifisert splitting, slik at klassefordelingen er lik i begge settene. For hvert opptak beregnes frekvensbånds-energier fra L- og H-signalene via rFFT med Hanning-vindusvekting. CNN normaliserer per opptak ved å dele på absolutt maksimum. MLP normaliserer med RobustScaler tilpasset eksklusivt til treningssettet. Siden det er flere eksempler av noen droner enn andre, altså en klasseubalanse, har vi gitt de sjeldne dronene høyere vekt under treningen. Dette gjør at modellene lærer like mye fra alle klassene, selv om de ikke har like mye data.
