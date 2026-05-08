= Introduksjon <intro>

Droner har blitt stadig mer tilgjengelige og brukes i dag til mange sivile og militære formål, blant annet inspeksjon, overvåking, beredskap og datainnsamling. Samtidig kan droner også utgjøre en sikkerhetsrisiko dersom de brukes i områder der de ikke skal være, eller dersom de benyttes til ulovlig overvåking, smugling eller forstyrrelse av kritisk infrastruktur. Det er derfor nyttig å kunne oppdage og analysere droneaktivitet automatisk.

I dette prosjektet undersøker vi om maskinlæringsmodeller kan bruke RF-signaler til å klassifisere hvilken modus en drone befinner seg i. Med dronemodus mener vi en mer detaljert klassifisering av aktiviteten eller tilstanden til dronen, ikke bare om en drone er til stede eller hvilken dronetype det er. Dette er interessant fordi modus kan gi mer operasjonell informasjon enn ren deteksjon. For eksempel kan det være relevant å skille mellom en drone som bare er påslått, en drone som flyr, og en drone som sender video.

== Bakgrunn

Tradisjonelle metoder for dronedeteksjon inkluderer radar, kamera og akustiske sensorer. Disse metodene kan være nyttige, men de har også begrensninger. Kameraer er avhengige av lys- og siktforhold, akustiske sensorer kan påvirkes av støy fra omgivelsene, og radarbaserte løsninger kan kreve dyrt eller spesialisert utstyr. RF-basert analyse er derfor et interessant alternativ, siden de fleste droner sender radiosignaler for styring, telemetri eller videooverføring.

DroneRF-datasettet inneholder radiofrekvens-opptak (RF-opptak) fra ulike droner og ulike aktivitetsmoduser. I prosjektets kode er datasettet strukturert med flere mulige klassifikasjonsmål: `target_binary` for drone mot bakgrunn, `target_family` for dronefamilie, og `target_mode` for en mer detaljert 10-klasse modusklassifisering. Hovedfokuset i prosjektet er `target_mode`, men vi bruker også informasjon om dronefamilie i CNN-eksperimentet for å undersøke om samtidig læring av type og modus kan gi nyttig representasjonslæring.

Rå RF-opptak er lange kontinuerlige signaler og kan ikke uten videre brukes direkte som input til en modell. Derfor må signalene deles inn i mindre segmenter eller vinduer og representeres på en form som gjør dem egnet for maskinlæring. For Multi-Layer Preception-modellen (MLP-modellen) brukes window-baserte features, mens Convolutional Netural Network-modell (CNN-modellen) bruker rå høy/lav (H/L)-signaler som en todimensjonal input med to kanaler. Dette gjør at modellene undersøker problemet på to forskjellige måter: MLP lærer fra ferdig konstruerte feature-vektorer, mens CNN forsøker å lære mønstre direkte fra signalstrukturen.

== Problemstilling

Prosjektet handler om hvorvidt RF-signaler kan brukes til å klassifisere dronemodus ved hjelp av maskinlæring. Vi avgrenser dermed hoveddelen av prosjektet til å undersøke aktivitet eller tilstand, heller enn bare å klassifisere dronetype. Dette valget er interessant fordi modus kan si mer om hva dronen faktisk gjør.

Samtidig er dette en krevende klassifikasjonsoppgave. Forskjellene mellom enkelte moduser kan være små, og RF-signaturene kan påvirkes av både dronetype, kommunikasjonsteknologi, støy og hvordan signalene segmenteres. En viktig del av prosjektet er derfor ikke bare å undersøke hvor høy nøyaktighet modellene oppnår, men også hvilke klasser de forveksler.

Basert på disse parameterne er det utarbeidet følgende problemstilling:

#set quote(block: true)
#quote()[_Kan maskinlæringsmodeller klassifisere dronemodus basert på RF-signaler fra DroneRF-datasettet?_]

I prosjektet sammenligner vi én MLP- og én CNN-modell. MLP-modellen kan trenes direkte på valgt mål, inkludert modusklassifisering, mens CNN-modellen er bygget som en multitask-modell som predikerer både dronefamilie og dronemodus. Dette gjør det mulig å undersøke om en modell som lærer begge oppgavene samtidig kan finne bedre signalrepresentasjoner enn en modell som bare lærer én target om gangen.

Modellene evalueres med nøyaktighet, macro-F1, presisjon, recall og confusion matrix. Macro-F1 er spesielt viktig fordi modusklassene kan være ubalanserte, og fordi total nøyaktighet alene kan gi et misvisende bilde dersom modellen hovedsakelig treffer de største klassene. Confusion matrix brukes for å se hvilke moduser modellene blander sammen, og for å diskutere om feilene skyldes dataene, feature-representasjonen eller modellarkitekturen.

=== Forskningsspørsmål

For å svare på problemstillingen undersøkes følgende forskningsspørsmål:

1. Hvor godt klassifiserer en MLP-modell dronemodus når den trenes på window-baserte features?
2. Hvor godt klassifiserer en CNN-modell dronemodus når den trenes på rå H/L RF-signaler?
3. Gir multitask-læring, der modellen predikerer både dronefamilie og dronemodus, bedre grunnlag for modusklassifisering?
4. Hvilke dronemoduser blir oftest forvekslet, og kan feilene forklares med likheter i RF-signalene eller begrensninger i datasettet?


