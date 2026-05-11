= Introduksjon <intro>

Droner brukes i dag til mange sivile og militære formål, noe som kan utgjøre en sikkerhetsrisiko. Det er derfor nyttig å kunne oppdage og analysere droneaktivitet automatisk. RF-basert analyse er et interessant alternativ til kamera, akustiske sensorer og radar, siden de fleste droner sender radiosignaler for styring og videooverføring.

I dette prosjektet undersøkes det om maskinlæringsmodeller kan bruke RF-signaler til å klassifisere hvilken modus en drone befinner seg i. Dronemodus gir mer operasjonell informasjon enn ren deteksjon — Blant annet skilles mellom en drone som er påslått, en som flyr, og en som sender video.

== Bakgrunn

Tradisjonelle metoder for dronedeteksjon inkluderer radar, kamera og akustiske sensorer. Kameraer er avhengige av lys- og siktforhold, akustiske sensorer påvirkes av støy, og radarbaserte løsninger kan kreve dyrt, spesialisert utstyr. RF-basert analyse er et interessant alternativ, siden de fleste droner sender radiosignaler for styring, telemetri og videooverføring. Disse signalene er til stede uavhengig av siktforhold og kan potensielt gi informasjon om hva dronen gjør, ikke bare om den er i lufta.

DroneRF-datasettet @DroneRF_dataset inneholder RF-opptak fra tre kommersielle dronemodeller i ulike aktivitetstilstander. Rå RF-signaler er lange tidsserier som ikke kan brukes direkte som modellinput. Signalene må segmenteres i vinduer og representeres på en form som egner seg for maskinlæring. Prosjektet undersøker to representasjonsstrategier og modeller: en MLP som opererer på sammensatte frekvensbånds-statistikker, og en CNN som arbeider direkte på vinduesekvenser. MLP ble valgt fordi den er en veletablert modell for tabelldata og gir et naturlig sammenligningspunkt. CNN ble valgt fordi den er kjent for å fange lokale mønstre i sekvensielle data uten å kreve manuell feature-engineering.

== Problemstilling

_Kan maskinlæringsmodeller klassifisere dronemodus basert på RF-signaler?_

Modellene evalueres med nøyaktighet, macro-F1, presisjon, recall og confusion matrix. Macro-F1 er spesielt viktig fordi modusklassene er ubalanserte — høy nøyaktighet alene kan gi et misvisende bilde dersom modellen favoriserer de største klassene. Confusion matrix brukes for å identifisere hvilke moduser som forveksles, og om feilmønsteret kan forklares med likheter i RF-signaturene eller begrensninger i datasettstrukturen.

=== Forskningsspørsmål

1. Hvor godt klassifiserer en MLP-modell dronemodus på sammensatte frekvens-statistikker?
2. Hvor godt klassifiserer en CNN-modell dronemodus på vinduesekvenser av frekvensdomenet?
3. Hvilke dronemoduser forveksles oftest, og kan feilene forklares med datasettet eller signallikhet?

// Er dette det vi faktisk gjør
/// Whaatt