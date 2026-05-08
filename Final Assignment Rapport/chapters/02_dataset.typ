= Dataset <Teori>
I dette prosjektet bruker vi DroneRF-datasettet, som består av RF-opptak fra kommersielle droner under kontrollerte forhold @DroneRF_dataset. Datasettet inneholder opptak fra tre dronetyper: Parrot Bebop, Parrot AR Drone og DJI Phantom, samt bakgrunnsopptak uten aktiv drone. Klasseubalansen er moderat — den største klassen (63 opptak) er om lag 1,6 ganger større enn den minste (39 opptak) — og evaluering gjøres derfor med macro-F1 i tillegg til nøyaktighet.

== Utvelging av features fra datasettet
Hvert opptak er knyttet til en BUI-kode der de tre første bitene identifiserer maskinvaren og de to siste beskriver dronens modus:

  - Modus 1 (00): Påslått og tilkoblet kontroller.
  - Modus 2 (01): Automatisk sveveflyging.
  - Modus 3 (10): Flyging uten videoopptak.
  - Modus 4 (11): Flyging med videoopptak.

#image("../images/bui.png")

Dronene kommuniserer over WiFi#sym.trademark på 2,4 GHz med en total båndbredde på opptil 80 MHz @DroneRF_dataset. Siden én mottaker hadde 40 MHz øyeblikkelig båndbredde ble to mottakere benyttet parallelt — én for lavt bånd (L) og én for høyt bånd (H). Med 40 MHz samplingsfrekvens og 10 000 000 verdier per fil representerer én fil 0,25 sekunder.

@time_11000L_0 og @time_11000H_0 viser råsignalet for et eksempelopptak.

#figure(
  image("../images/11000L_0.png"),
  caption: [11000L_0.csv i tidsdomenet]
)<time_11000L_0>
#figure(
  image("../images/11000H_0.png"),
  caption: [11000H_0.csv i tidsdomenet]
)<time_11000H_0>

Frekvensinnholdet er mer stabilt enn tidsdomenet, siden dronene kommuniserer på faste frekvenser. Siden CSV-filene kun inneholder den reelle delen av signalet, gir rFFT et basebånd-spekter fra 0 til 20 MHz — frekvensinformasjonen er bevart konsistent på tvers av alle opptak, noe som er tilstrekkelig for klassifisering. @freq_1000L_0 viser sammenligningen mellom tid- og frekvensdomenet.

#figure(
  image("../img/11000L_0_freq.png", width: 90%),
  caption: [Tids- og frekvensdomene for 11000L_0]
)<freq_1000L_0>

Begge modellene bruker glidende vinduisering: 64 000 sampler per vindu, 50 % overlapping, og 32 frekvensbånd per kanal via FFT. L- og H-båndet gir 64 frekvensbånds-energier per vindu, og alle opptak paddes eller avkortes til 300 vinduer. CNN mottar disse direkte som en (300, 64)-matrise. MLP komprimerer dem til 192 egenskaper ved å beregne gjennomsnitt, standardavvik og maksimum per frekvensbånd.

== Eksplorativ Dataanalyse (EDA)

Klassefordelingen er vist i @class_distribution_table. Klassene 2 og 3 slår sammen opptak fra ulike dronefamilier under samme BUI-kode, noe som øker variasjonen innad i disse klassene og gjør dem vanskeligere å skille.

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

Datasettet ble delt 80/20 i trenings- og testsett med stratifisert splitting, slik at klassefordelingen er lik i begge settene. For hvert opptak beregnes frekvensbånds-energier fra L- og H-signalene via rFFT med Hanning-vindusvekting. CNN normaliserer per opptak ved å dele på absolutt maksimum. MLP normaliserer med RobustScaler tilpasset eksklusivt til treningssettet. Klasseubalansen håndteres ved frekvensbasert prøvevekting under trening for begge modeller.
