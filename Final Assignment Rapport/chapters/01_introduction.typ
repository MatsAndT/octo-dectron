= Introduksjon <intro>

Droner brukes i dag til mange sivile og militære formål, men kan også utgjøre en sikkerhetsrisiko ved ulovlig bruk. Det er derfor nyttig å kunne oppdage og analysere droneaktivitet automatisk. RF-basert analyse er et interessant alternativ til kamera, akustiske sensorer og radar, siden de fleste droner sender radiosignaler for styring og videooverføring.

I dette prosjektet undersøker vi om maskinlæringsmodeller kan bruke RF-signaler til å klassifisere hvilken modus en drone befinner seg i. Dronemodus gir mer operasjonell informasjon enn ren deteksjon — det skiller mellom en drone som er påslått, en som flyr, og en som sender video.

== Bakgrunn

DroneRF-datasettet inneholder RF-opptak fra ulike droner og aktivitetsmoduser @DroneRF_dataset. Råsignalene deles inn i overlappende vinduer, og for hvert vindu beregnes gjennomsnittlig energi per frekvensbånd. For MLP komprimeres vinduene til én flat feature-vektor per opptak. CNN mottar vindusekvensen direkte og kan finne mønstre i hvordan frekvensprofilen endrer seg gjennom opptaket.

== Problemstilling

#set quote(block: true)
#quote()[_Kan maskinlæringsmodeller klassifisere dronemodus basert på RF-signaler fra DroneRF-datasettet?_]

Modellene evalueres med nøyaktighet, macro-F1 og confusion matrix. Macro-F1 er viktig fordi klassene er ubalanserte, og confusion matrix viser hvilke moduser som forveksles.

=== Forskningsspørsmål

1. Hvor godt klassifiserer en MLP-modell dronemodus på aggregerte frekvensbånds-energier?
2. Hvor godt klassifiserer en CNN-modell dronemodus på vinduesekvenser av frekvensbånds-energier?
3. Hvilke dronemoduser forveksles oftest, og kan feilene forklares med datasettet eller signallikhet?
