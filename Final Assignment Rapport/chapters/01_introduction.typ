= Introduksjon <intro>

Droner brukes i dag til mange sivile og militære formål, men kan også utgjøre en sikkerhetsrisiko ved ulovlig bruk. Det er derfor nyttig å kunne oppdage og analysere droneaktivitet automatisk. RF-basert analyse er et interessant alternativ til kamera, akustiske sensorer og radar, siden de fleste droner sender radiosignaler for styring og videooverføring.

I dette prosjektet undersøker vi om maskinlæringsmodeller kan bruke RF-signaler til å klassifisere hvilken modus en drone befinner seg i. Dronemodus gir mer operasjonell informasjon enn ren deteksjon — det skiller mellom en drone som er påslått, en som flyr, og en som sender video.

== Bakgrunn

Tradisjonelle metoder for dronedeteksjon inkluderer radar, kamera og akustiske sensorer. Kameraer er avhengige av lys- og siktforhold, akustiske sensorer påvirkes av støy, og radarbaserte løsninger kan kreve spesialisert utstyr. RF-basert analyse er et interessant alternativ, siden de fleste droner sender radiosignaler for styring, telemetri og videooverføring. Disse signalene er til stede uavhengig av siktforhold og kan potensielt gi informasjon om hva dronen gjør, ikke bare om den er i lufta.

DroneRF-datasettet @DroneRF_dataset inneholder RF-opptak fra tre kommersielle dronemodeller i ulike aktivitetstilstander, og er én av få offentlig tilgjengelige databaser for dette formålet. Rå RF-signaler er lange tidsserier som er uegnet direkte som modellinput. Signalene må representeres på en form som egner seg for maskinlæring. Prosjektet undersøker to representasjonsstrategier: aggregerte frekvensbånds-statistikker for MLP, og direkte vinduesekvenser for CNN.


== Problemstilling

#set quote(block: true)
#quote()[_Kan maskinlæringsmodeller klassifisere dronemodus basert på RF-signaler fra DroneRF-datasettet?_]

Modellene evalueres med nøyaktighet, macro-F1, presisjon, recall og confusion matrix. Macro-F1 er spesielt viktig fordi modusklassene er ubalanserte — høy nøyaktighet alene kan gi et misvisende bilde dersom modellen favoriserer de største klassene. Confusion matrix brukes for å identifisere hvilke moduser som forveksles, og om feilmønsteret kan forklares med likheter i RF-signaturene eller begrensninger i datasettstrukturen.

=== Forskningsspørsmål

1. Hvor godt klassifiserer en MLP-modell dronemodus på sammensatte frekvens-statistikker?
2. Hvor godt klassifiserer en CNN-modell dronemodus på vinduesekvenser av frekvensdomenet?
3. Hvilke dronemoduser forveksles oftest, og kan feilene forklares med datasettet eller signallikhet?
