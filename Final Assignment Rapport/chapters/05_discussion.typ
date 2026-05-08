= Drøfting 

== Valg av klasser

Et av de tidlige valgene vi tok var om vi skulle trene modellen til å klassifisere datasettet basert på dronetype eller dronemodus. Vi valgte å klassifisere basert på dronemodus, det vil si tilstander som påslått og tilkoblet eller automatisk sveveflyging. For å ta dette valget måtte vi spørre oss selv hva som ville være mest nyttig i et reelt scenario. Vi konkluderte raskt med at klassifisering basert på dronemodell ikke ga særlig mening. Datasettet inneholder kun tre dronemodeller, mens det i virkeligheten finnes langt flere. I tillegg er det ikke uvanlig å bygge droner selv. Basert på dataene vi har tilgjengelig gir det derfor mest mening å klassifisere ut fra hva dronen faktisk gjør.

Spørsmålet som oppstår etter dette valget er om RF-signalene varierer mellom modeller. Er det mulig å klassifisere dronemodus uten å kjenne til dronemodellen? #highlight[(Dette hadde jeg ikke svaret på når jeg skrev, så det må fylles inn!)]

Én måte å lage en mer robust modell på i fremtiden ville være å klassifisere både modell og modus. På den måten kunne man først identifisere hvilken type drone det er snakk om, og deretter klassifisere hvilken modus den befinner seg i. Dette ville imidlertid kreve langt mer data fra flere dronemodeller. Man kan også argumentere for at RF-signalene fra ulike modeller ikke nødvendigvis er veldig forskjellige, siden teknologier som videooverføring ofte benytter standardiserte protokoller. #highlight[Burde sikkert finne en kilde her?]



== Prestasjon MLP

Det faktum at testnøyaktigheten stagnerte på rundt 20–30 % til tross for omfattende hyperparameter-testing, indikerer at hovedutfordringen ligger i selve datagrunnlaget fremfor modellens arkitektur. Den lave variansen i mange av de valgte egenskapene tyder på at de inneholdt begrenset informasjon, noe som gjorde det vanskelig for MLP-algoritmen å etablere pålitelige beslutningsgrenser mellom de ulike modusene.

En fundamental årsak til den lave prestasjonen er sannsynligvis beslutningen om å klassifisere dronens modus på tvers av ulike produsenter. Siden RF-signaler styres av produsentspesifikke protokoller, er det ingen teknisk nødvendighet for at like moduser skal ha sammenfallende signaturer når de kommer fra forskjellige kilder. Prosjektgruppen valgte opprinnelig denne tilnærmingen av operative hensyn, da modusen gir viktig situasjonsforståelse. I retrospekt er det imidlertid grunn til å anta at det ville vært en betydelig enklere oppgave å identifisere spesifikke dronemodeller, ettersom disse trolig har mer distinkte og konsistente RF-karakteristikker enn de generiske modusene.

Datakvaliteten kan også ha vært begrenset av segmenteringen i korte tidsvinduer på 0,25 sekunder. Dette kan ha vært utilstrekkelig for å fange opp de dynamiske endringene i RF-spekteret som definerer en modus over tid. Selv om det totale datavolumet var stort, var antallet unike opptak per modus begrenset, noe som økte risikoen for at modellen lærte seg kjennetegn ved spesifikke måleperioder fremfor generelle modustrekk. Begrenset tid for prosjektgruppen forhindret videre utforskning av mer komplekse egenskaper eller en omlegging til drone-identifisering, som trolig ville vært nødvendige grep for å øke treffsikkerheten til MLP-modellen.



  ==



== 
