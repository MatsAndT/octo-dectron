= Resultat <Labtester>

== MLP

Eksperimentene med Multi-Layer Perceptron (MLP) ble utført på et datasett bestående av 227 opptak fordelt over fem dronemoduser (0–4), med en klassefordeling på henholdsvis 63, 41, 42, 42 og 39 opptak per klasse. Hvert opptak ble representert som en flat feature-vektor med 192 egenskaper, beregnet ved å sammenstille 300 frekvensbånds-vinduer med statistiske verdier (gjennomsnitt, standardavvik og maksimum per band). For å etablere en grunnlinje ble det først kjørt en Dummy Classifier basert på mest-hyppig-strategi, som oppnådde en testnøyaktighet på 28,26 % og en macro-F1 på 0,09. En uregulert "Kitchen Sink"-modell viste en treningsnøyaktighet på 100 %, men falt til 84,78 % på testsettet, noe som indikerer at modellen memorerte treningsdataene fremfor å generalisere.

Ved bruk av GridSearchCV med 5-fold stratifisert kryssvalidering og macro-F1 som scoringsmetrikk ble den beste modellen identifisert med ett skjult lag med 64 nevroner, regulariseringsparameter α = 1,0 og læringsrate 0,001. Denne konfigurasjonen oppnådde en CV macro-F1 på 0,85, en treningsnøyaktighet på 98,90 % og en testnøyaktighet på 82,61 %, med en test macro-F1 på 0,82, som også kan indikere en form for overfitting. @tab-mlp-report oppsummerer per-klasse-ytelsen:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: center,
    table.header(
      [*Klasse*], [*Presisjon*], [*Recall*], [*F1-score*], [*Støtte*],
    ),
    [0], [1,00], [0,69], [0,82], [13],
    [1], [0,80], [1,00], [0,89], [8],
    [2], [0,67], [0,89], [0,76], [9],
    [3], [0,89], [1,00], [0,94], [8],
    [4], [0,83], [0,62], [0,71], [8],
    [*Macro avg*], [*0,84*], [*0,84*], [*0,82*], [*46*],
    [*Weighted avg*], [*0,85*], [*0,83*], [*0,82*], [*46*],
  ),
  caption: [Klassifikasjonsrapport for MLP-modellen på testsettet (N = 46).]
) <tab-mlp-report>

Analyse av forvirringsmatrisen i @MLP_confusion gir en visuell fremstilling av modellens tendenser. Diagonalen er tydelig dominerende, og modellen klassifiserer modus 1 og modus 3 med perfekt recall. De største utfordringene finnes i modus 0, der 4 av 13 opptak feiltolkes som andre klasser, og modus 4, der 3 av 8 opptak klassifiseres feil. Modus 2 oppnår god recall (8 av 9), men noe lavere presisjon, noe som indikerer at modellen iblant forveksler andre klasser med modus 2.

#figure(
  caption: [Forvirringsmatrise for MLP],
  image("../img/mlp v2 confusion matrix.png", width: 80%)
)<MLP_confusion>

== CNN

Eksperimentene med CNN ble gjennomført på det samme datasettet som MLP, bestående av 227 opptak fordelt over fem dronemoduser. I motsetning til MLP, som opererer på manuelt konstruerte feature-vektorer, ble CNN-modellen trent direkte på frekvensbåndet hentet via glidende vinduing av råsignalene. Hvert opptak ble representert som en matrise av form (300, 64), der 300 tilsvarer antall vinduer på 64 000 sampler med 50 % overlapping, og 64 tilsvarer 32 frekvensbånd fra henholdsvis lavt og høyt frekvensbånd.

Som referansepunkt ble det innledningsvis testet en modell med for høy kapasitet — 203 461 parametere fordelt over tre konvolusjonsblokker med 64, 128 og 256 filtre. Denne modellen nådde en treningsnøyaktighet på om lag 89 %, mens valideringsnøyaktigheten stagnerte på 35 % og falt videre ved fortsatt trening. Forvirringsmatrisen viste at modellen hadde kollapset til å predikere nesten utelukkende én klasse. Dette bekreftet at modellens kapasitet var langt høyere enn det 181 treningsfiler kan bære.

=== Testprosedyre

Den endelige modellen er bevisst holdt liten (ca. 11 000 parametere) for å unngå at den bare "pugget" treningsdataene. Arkitekturen består av to lag som leter etter mønstre i signalene, etterfulgt av flere sikkerhetsmekanismer (L2-straff, dropout og kunstig støy) som tvinger modellen til å lære generelle kjennetegn fremfor uvesentlige detaljer.

Selve treningen ble styrt av følgende grep:

    - For å håndtere ubalanse i datasettet ble det benyttet klassevektlegging. Dette sikrer at underrepresenterte dronemoduser får økt betydning under treningen, slik at modellen lærer å identifisere alle klasser like effektivt uavhengig av antall observasjoner.

    - Smart avslutning: Treningen stoppet automatisk hvis modellen sluttet å forbedre seg på nye data (Early Stopping), og vi beholdt den aller beste versjonen av modellen.

    - Evaluering: Modellen ble testet på 20 % av dataene (46 opptak) som den aldri hadde sett før. Vi sørget for at denne testgruppen hadde nøyaktig samme fordeling av dronetyper som resten av settet (stratifisert splitting) for å få et ærlig svar på hvor god modellen faktisk er.

=== Resultater

// ↓ LEGG INN learning_curves.png HER ↓
#figure(
  image("../img/cnn loss accuracy.png", width: 100%),
  caption: [Trenings- og valideringstap (venstre) og nøyaktighet (høyre) per epoke for CNN-modellen.]
) <fig-cnn-curves>

@fig-cnn-curves viser trenings- og valideringsforløpet over 200 epoker. Begge tapskurvene synker konsistent gjennom hele treningsforløpet uten at de divergerer, fra henholdsvis 2,4 og 2,0 i starten til om lag 0,75–0,90 ved slutten. Nøyaktighetskurvene følger hverandre tett med moderat støy, noe som er forventet ved en batchstørrelse på 16 over et lite datasett. Det begrensede gapet mellom trenings- og valideringskurver indikerer at regulariseringsstrategien — L2, dropout og GaussianNoise — var effektiv i å hindre modellen fra å memorere treningsdataene.

Den endelige modellen oppnådde en testnøyaktighet på 71,7 % (33 av 46 korrekt klassifiserte opptak). @tab-cnn-report oppsummerer per-klasse-ytelsen:

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

Macro-F1 på 0,66 er et mer representativt mål enn total nøyaktighet gitt klasseubalansen, og overstiger klart det en tilfeldig klassifiserer ville oppnå. MLP-modellen oppnådde 82,61 % testnøyaktighet og macro-F1 på 0,82 på de samme dataene, noe som er noe høyere enn CNN.

=== Begrensninger

En vesentlig metodisk begrensning er at testsettet kun består av 46 opptak. Med så få testpunkter er estimatene for presisjon, recall og F1 per klasse statistisk usikre; for klasser med åtte til ni testeksempler vil ett enkelt feilklassifisert opptak gi et utslag på over ti prosentpoeng i recall. Resultatene bør derfor tolkes som en indikasjon på modellens generaliseringsevne heller enn som presise ytelsesestimater.
