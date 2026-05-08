= Resultat <Labtester>

== MLP

Eksperimentene med Multi-Layer Perceptron (MLP) ble utført på et datasett bestående av 227 prøver og 28 initielle egenskaper, fordelt over fem dronemoduser (0–4). For å etablere en grunnlinje ble det først kjørt en Dummy Classifier, som oppnådde en testnøyaktighet på 28,26 %. En uregulert "Kitchen Sink"-modell viste en treningsnøyaktighet på 100 %, men falt til 19,57 % på testsettet, noe som indikerer at modellen i sin råform memorerte støy fremfor reelle mønstre.

Gjennom en filtreringsprosess ble redundante egenskaper som fL_max og fH_max, samt fH_min på grunn av lav varians, fjernet. Dette etterlot 25 egenskaper for optimalisering. Ved bruk av GridSearchCV ble den beste modellen identifisert med en arkitektur på to lag med 16 nevroner hver (16, 16), en læringsrate på 0,001 og en regulariseringsparameter (alpha) på 0,05. Denne konfigurasjonen stabiliserte modellen, med en treningsnøyaktighet på 38,67 % og en testnøyaktighet på 26,09 %. Resultatet viser en modell som i større grad forsøker å generalisere, selv om treffprosenten forblir lav.

Analyse av forvirringsmatrisen i @MLP_confusion gir en visuell fremstilling av modellens tendenser. Matrisen viser en tydelig skjevhet mot modus 3, som predikeres betydelig oftere enn det faktiske antallet i testsettet. Det observeres også en markant overlapping mellom modus 0 og modus 2, hvor modellen ofte forveksler disse to klassene. Den spredte distribusjonen av tall utenfor diagonalen i @MLP_confusion dokumenterer at de nåværende egenskapene ikke gir tilstrekkelig separasjon mellom klassene i det valgte egenskapsrommet.

#figure(
  caption: [Forvirringsmatrise for MLP],
  image("../img/MLP_confusion.png", width: 80%)
)<MLP_confusion>


=== Testprosedyre



=== Resultater



=== Begrensninger 



== Eksperiment 2



=== Testprosedyre



=== Resultater



== Eksperiment 3



=== Testprosedyre



=== Resultater

== CNN

Eksperimentene med CNN ble gjennomført på det samme datasettet som MLP, bestående av 227 opptak fordelt over fem dronemoduser (0–4), med en klassefordeling på henholdsvis 63, 41, 42, 42 og 39 opptak per klasse. I motsetning til MLP, som opererer på manuelt konstruerte feature-vektorer, ble CNN-modellen trent direkte på frekvensbånd-energier ekstrahert via glidende vinduing av råsignalene. Hvert opptak ble representert som en matrise av form (300, 64), der 300 tilsvarer antall vinduer på 64 000 sampler med 50 % overlapping, og 64 tilsvarer 32 frekvensbånd fra henholdsvis lavt og høyt frekvensbånd.

Som referansepunkt ble det innledningsvis testet en modell med for høy kapasitet — 203 461 parametere fordelt over tre konvolusjonsblokker med 64, 128 og 256 filtre. Denne modellen nådde en treningsnøyaktighet på om lag 89 %, mens valideringsnøyaktigheten stagnerte på 35 % og falt videre ved fortsatt trening. Forvirringsmatrisen viste at modellen hadde kollapset til å predikere nesten utelukkende én klasse. Dette bekreftet at modellens kapasitet var langt høyere enn det 181 treningsfiler kan bære, et fenomen som er analogt med «Kitchen Sink»-kollapsen observert for MLP.

=== Testprosedyre

Den endelige CNN-modellen ble redusert til om lag 11 000 parametere, fordelt over to konvolusjonsblokker med henholdsvis 16 og 32 filtre. Regularisering ble ivaretatt gjennom L2-straff (α = 0,005) på alle trenbare lag, dropout-rater på 0,4 etter hvert konvolusjonsblokk og 0,5 etter det fullt tilkoblede laget, samt et GaussianNoise-lag (σ = 0,05) som innebygd treningsaugmentering. Treningsprosedyren benyttet Adam-optimalisatoren med læringsrate 0,0005 og batchstørrelse 16. Frekvensbasert klassevekting ble benyttet for å kompensere for ubalansen mellom modusene. Tidlig stopp med tålmodighetsparameter 20 overvåket valideringstapet, og beste modellvekter ble restaurert ved treningsslutt. Modellen ble evaluert på et testsett bestående av 46 opptak, utgjørende 20 % av datasettet, partisjonert med stratifisert splitting for å bevare klassefordelingen.

=== Resultater

// ↓ LEGG INN learning_curves.png HER ↓
#figure(
  image("../img/cnn loss accuracy.png", width: 100%),
  caption: [Trenings- og valideringstap (venstre) og nøyaktighet (høyre) per epoke for CNN-modellen.]
) <fig-cnn-curves>

@fig-cnn-curves viser trenings- og valideringsforløpet over 185 epoker. Begge tapskurvene synker konsistent gjennom hele treningsforløpet uten at de divergerer, fra henholdsvis 2,4 og 2,0 i starten til om lag 0,75–0,90 ved slutten. Nøyaktighetskurvene følger hverandre tett med moderat støy, noe som er forventet ved en batchstørrelse på 16 over et lite datasett. Det begrensede gapet mellom trenings- og valideringskurver indikerer at regulariseringsstrategien — L2, dropout og GaussianNoise — var effektiv i å hindre modellen fra å memorere treningsdataene.

Den endelige modellen oppnådde en testnøyaktighet på 71,7 % (33 av 46 korrekt klassifiserte opptak). Tabell X oppsummerer per-klasse-ytelsen:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center,
    table.header(
      [*Klasse*], [*Presisjon*], [*Recall*], [*F1-score*], [*Støtte*],
    ),
    [0], [0,62], [1,00], [0,76], [13],
    [1], [1,00], [1,00], [1,00], [8],
    [2], [1,00], [0,22], [0,36], [9],
    [3], [1,00], [0,25], [0,40], [8],
    [4], [0,62], [1,00], [0,76], [8],
    [*Macro avg*], [*0,85*], [*0,69*], [*0,66*], [*46*],
    [*Weighted avg*], [*0,83*], [*0,72*], [*0,66*], [*46*],
  ),
  caption: [Klassifikasjonsrapport for CNN-modellen på testsettet (N = 46).]
) <tab-cnn-report>

// ↓ LEGG INN confusion_matrix.png HER ↓
#figure(
  image("../img/cnn confusion matrix.png", width: 70%),
  caption: [Forvirringsmatrise for CNN-modellen på testsettet.]
) <fig-cnn-confusion>

@fig-cnn-confusion viser forvirringsmatrisen for testsettet. Modellen klassifiserer modus 0 og modus 1 perfekt (henholdsvis 13/13 og 8/8 korrekte), og modus 4 med full recall (8/8). De største utfordringene finnes i modus 2 og modus 3, der modellen kun identifiserer 2 av 9 og 2 av 8 korrekt. For modus 2 klassifiseres 6 av 9 opptak feilaktig som modus 0, mens modus 3 fordeles mellom modus 0 (2 opptak) og modus 4 (4 opptak). Dette mønsteret, der modusene 2 og 3 forveksles med andre klasser fremfor med hverandre, tyder på at disse RF-signaturene deler kjennetegn med bakgrunnsaktivitet og flygesignaler fra andre moduser, heller enn at de er innbyrdes vanskelige å skille.

Macro-F1 på 0,66 er et mer representativt mål enn total nøyaktighet gitt klasseubalansen, og overstiger klart det en tilfeldig klassifiserer ville oppnå. Sammenlignet med MLP, som oppnådde en testnøyaktighet på 26,09 % med macro-F1 på tilsvarende lavt nivå, representerer CNN-modellens 71,7 % en vesentlig forbedring, og indikerer at den tidsseriebaserte, frekvensoppdelte representasjonen bærer betydelig mer diskriminerende informasjon enn de manuelt konstruerte feature-vektorene benyttet av MLP.

=== Begrensninger

En vesentlig metodisk begrensning er at testsettet kun består av 46 opptak. Med så få testpunkter er estimatene for presisjon, recall og F1 per klasse statistisk usikre; for klasser med åtte til ni testeksempler vil ett enkelt feilklassifisert opptak gi et utslag på over ti prosentpoeng i recall. Resultatene bør derfor tolkes som en indikasjon på modellens generaliseringsevne heller enn som presise ytelsesestimater.

En ytterligere begrensning er at de fem klassene i SimpleModel-implementasjonen er definert ut fra de to siste bitene i BUI-koden, noe som innebærer at moduser fra ulike dronefamilier — eksempelvis Bebop og AR Drone i modus 00 — samles i samme klasse. Denne forenklingen øker intra-klasse-variansen og gjør klassifikasjonsoppgaven vanskeligere enn om modusene var separert per dronefamilie. De to klassene med lav recall (modus 2 og 3) tilsvarer nettopp de modusene der to ulike dronefamilier er slått sammen, noe som understøtter denne hypotesen.
