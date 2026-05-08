= Drøfting 

== Valg av klasser

Et av de tidlige valgene vi tok var om vi skulle trene modellen til å klassifisere datasettet basert på dronetype eller dronemodus. Vi valgte å klassifisere basert på dronemodus, det vil si tilstander som påslått og tilkoblet eller automatisk sveveflyging. For å ta dette valget måtte vi spørre oss selv hva som ville være mest nyttig i et reelt scenario. Vi konkluderte raskt med at klassifisering basert på dronemodell ikke ga særlig mening. Datasettet inneholder kun tre dronemodeller, mens det i virkeligheten finnes langt flere. I tillegg er det ikke uvanlig å bygge droner selv. Basert på dataene vi har tilgjengelig gir det derfor mest mening å klassifisere ut fra hva dronen faktisk gjør.

Spørsmålet som oppstår etter dette valget er om RF-signalene varierer mellom modeller. Er det mulig å klassifisere dronemodus uten å kjenne til dronemodellen? #highlight[(Dette hadde jeg ikke svaret på når jeg skrev, så det må fylles inn!)]

Én måte å lage en mer robust modell på i fremtiden ville være å klassifisere både modell og modus. På den måten kunne man først identifisere hvilken type drone det er snakk om, og deretter klassifisere hvilken modus den befinner seg i. Dette ville imidlertid kreve langt mer data fra flere dronemodeller. Man kan også argumentere for at RF-signalene fra ulike modeller ikke nødvendigvis er veldig forskjellige, siden teknologier som videooverføring ofte benytter standardiserte protokoller. #highlight[Burde sikkert finne en kilde her?]



== Prestasjon MLP

MLP-modellen oppnådde en testnøyaktighet på 82,61 % og en macro-F1 på 0,82. Den viktigste faktoren for dette resultatet er signalrepresentasjonen: ved å aggregere 300 overlappende vinduer per opptak til 192 egenskaper (gjennomsnitt, standardavvik og maksimum per frekvensbånd) får modellen tilgang til informasjon om hvordan signalet varierer gjennom opptaket. Standardavviket fanger om aktiviteten er stabil eller skiftende fra vindu til vindu, og maksimumverdien bevarer informasjon om toppenergi som et enkelt gjennomsnitt ville skjult.

Et interessant funn er at modellen med den sterkeste regulariseringen (α = 1,0) og den enkleste arkitekturen (ett lag med 64 nevroner) slo alle større alternativer. Med kun 181 treningsfiler vil en modell med for mange frihetsgrader memorere spesifikke opptak fremfor å generalisere. GridSearchCV med macro-F1 som mål bekreftet dette: søket favoriserte konsekvent de minste og mest regulariserte konfigurasjonene.

MLP presterer noe bedre enn CNN (82,6 % vs 71,7 %) til tross for at CNN behandler vinduesekvensen direkte og ikke komprimerer den til tre statistikker per bånd. En mulig forklaring er at gjennomsnitt, standardavvik og maksimum per frekvensbånd fanger opp det meste av den nyttige informasjonen for dette datasettet, og at mer detaljert strukturinformasjon ikke tilfører noe gitt det lille treningssettet. Begge modellene sliter med de samme klassene — modus 2 og 3 — noe som tyder på at problemet ligger i dataene, ikke i valg av modelltype.



== Prestasjon CNN

CNN-modellen oppnådde en testnøyaktighet på 71,7 % og en macro-F1 på 0,66, noe som representerer en markant forbedring sammenlignet med MLP. Spørsmålet er hva som driver denne forskjellen, og hvorfor modellen likevel mislykkes med bestemte klasser.

Den viktigste faktoren for CNN-modellens ytelse er at den mottar signalet som en sekvens av 300 frekvensbånds-energivinduer og kan dermed finne mønstre i hvordan frekvensprofilen endrer seg gjennom opptaket. At modellen klarer 71,7 % med kun 11 000 parametere tyder på at RF-signalet faktisk inneholder strukturinformasjon utover et enkelt gjennomsnittsspekter.

Modellkapasitet viste seg å være en avgjørende variabel. Det første forsøket med 203 461 parametere kollapset til én-klasse-prediksjon, analogt med Kitchen Sink-kollapsen for MLP. Med kun 181 treningsfiler er det matematisk sett et svært lavt antall parameteroppdateringer per treningsparameter, noe som betyr at en stor modell raskt memorerer støy fremfor å generalisere. Den endelige modellen på 11 000 parametere, kombinert med L2-straff på alle lag, dropout og GaussianNoise-augmentering, gav et trenings-/valideringsgap som læringsforløpet i @fig-cnn-curves bekrefter er lite. Dette viser at valg av riktig modellkapasitet, ikke bare regulariseringsteknikk, er det primære middelet mot overfitting ved små datasett.

Klasseimbalansen ble håndtert gjennom frekvensbaserte klassevekter, men resultatene viser at dette ikke var tilstrekkelig for alle klasser. Modus 2 og modus 3 oppnådde recall på henholdsvis 22 % og 25 %, mens modus 0, 1 og 4 ble klassifisert riktig. En sannsynlig forklaring er strukturen i BUI-koden: klasse 2 og 3 er definert av de to siste bitene og inneholder opptak fra to ulike dronefamilier, eksempelvis Bebop og AR Drone, i den samme klassen. Dersom RF-signaturen for «automatisk svev» varierer mellom produsenter, vil variasjonen innad i klasse 2 og 3 være grunnleggende høyere enn i klasser dominert av én dronefamilie. Forvirringsmatrisen i @fig-cnn-confusion støtter dette: modus 2 og 3 forveksles ikke primært med hverandre, men fordeles mellom modus 0 og 4, noe som tyder på at modellen lærer produsentspesifikke signaturer og ikke en modusspesifikk signatur som er stabil på tvers av produsenter.

Funnet underbygger det samme argumentet som ble reist under valg av klasser: klassifisering av dronemodus på tvers av produsenter er en vanskeligere oppgave enn klassifisering av dronefamilie, fordi RF-protokollene som definerer en modus ikke nødvendigvis er felles for ulike fabrikanter. Begge modellene støter på den samme grunnleggende utfordringen: de klassene som blander opptak fra ulike dronefamilier har for stor variasjon innad til at modellene klarer å lære en pålitelig felles signatur.

==
