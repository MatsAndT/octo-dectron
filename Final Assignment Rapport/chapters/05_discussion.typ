= Drøfting 

== Valg av klasser

Et av de tidlige valgene vi tok var om vi skulle trene modellen til å klassifisere datasettet basert på dronetype eller dronemodus. Vi valgte å klassifisere basert på dronemodus, det vil si tilstander som påslått og tilkoblet eller automatisk sveveflyging. For å ta dette valget måtte vi spørre oss selv hva som ville være mest nyttig i et reelt scenario. Vi konkluderte raskt med at klassifisering basert på dronemodell ikke ga særlig mening. Datasettet inneholder kun tre dronemodeller, mens det i virkeligheten finnes langt flere. I tillegg er det ikke uvanlig å bygge droner selv. Basert på dataene vi har tilgjengelig gir det derfor mest mening å klassifisere ut fra hva dronen faktisk gjør.

Spørsmålet som oppstår etter dette valget er om RF-signalene varierer mellom modeller. Er det mulig å klassifisere dronemodus uten å kjenne til dronemodellen? #highlight[(Dette hadde jeg ikke svaret på når jeg skrev, så det må fylles inn!)]

Én måte å lage en mer robust modell på i fremtiden ville være å klassifisere både modell og modus. På den måten kunne man først identifisere hvilken type drone det er snakk om, og deretter klassifisere hvilken modus den befinner seg i. Dette ville imidlertid kreve langt mer data fra flere dronemodeller. Man kan også argumentere for at RF-signalene fra ulike modeller ikke nødvendigvis er veldig forskjellige, siden teknologier som videooverføring ofte benytter standardiserte protokoller. #highlight[Burde sikkert finne en kilde her?]



== Prestasjon MLP

MLP-modellen oppnådde en testnøyaktighet på 82,61 % og en macro-F1 på 0,82, noe som er en vesentlig forbedring sammenlignet med den opprinnelige implementasjonen som stagnerte på rundt 26 %. Den primære driveren for forbedringen var ikke arkitektur eller hyperparametre, men representasjonen av inngangsdataene. Der den første MLP-versjonen opererte på 28 manuelt konstruerte egenskaper fra én enkelt FFT per opptak, benytter den oppdaterte modellen 192 statistisk poolede egenskaper fra 300 overlappende vinduer. Dette gir modellen tilgang til temporal variasjon i signalet — standardavviket per frekvensbånd fanger opp om aktiviteten er stabil eller skiftende, og maksimumverdien bevarer informasjon om toppenergi som et middelverdibasert sammendrag ville skjult.

Et interessant funn er at modellen med den sterkeste regulariseringen (α = 1,0) og den enkleste arkitekturen (ett lag med 64 nevroner) slo alle større alternativer. Dette er konsistent med det samme kapasitetsproblemet som ble observert for CNN: med 181 treningsfiler er antall unike treningspunkter sterkt begrenset, og en modell med for mange frihetsgrader vil memorere spesifikke opptak fremfor å generalisere. GridSearchCV med macro-F1-scoring bekreftet dette systematisk, da søket konsekvent favoriserte de minste og mest regulariserte konfigurasjonene.

Sammenlignet med CNN-modellen (71,7 % nøyaktighet, macro-F1 0,66) presterer MLP marginalt bedre på disse testdataene. Dette er i utgangspunktet overraskende, gitt at CNN behandler vinduesekvensen direkte og har tilgang til temporal struktur som MLP-poolingen reduserer til tre statistikker. En mulig forklaring er at de tre statistiske sammendragene — gjennomsnitt, standardavvik og maksimum per frekvensbånd — fanger opp den mest diskriminerende informasjonen for dette datasettet, og at den resterende temporale strukturen ikke tilfører vesentlig ny informasjon gitt datasettets størrelse. En annen forklaring er at CNN-modellens regularisering ikke var tilstrekkelig fininnstilt, og at en mer systematisk hyperparametersøk for CNN ville gitt tilsvarende resultater. Begge modellene sliter likevel med de samme klassene: modus 0 og modus 4 har lavest recall, noe som tyder på at utfordringen er datarelatert heller enn modellrelatert.



== Prestasjon CNN

CNN-modellen oppnådde en testnøyaktighet på 71,7 % og en macro-F1 på 0,66, noe som representerer en markant forbedring sammenlignet med MLP. Spørsmålet er hva som driver denne forskjellen, og hvorfor modellen likevel mislykkes med bestemte klasser.

Den mest sentrale faktoren er hvordan signalet representeres. MLP opererte på 28 manuelt konstruerte egenskaper beregnet fra én enkelt 4096-punkts FFT per opptak, noe som i praksis komprimerte hele opptaket til én statistisk øyeblikksverdi. CNN-modellen mottok derimot signalet som en sekvens av 300 frekvensbånds-energivinduer, der hvert vindu fanget den lokale spektrale tilstanden over 64 000 sampler med 50 % overlapping. Dette gir modellen tilgang til temporal struktur — endringer i frekvensinnholdet over tid — som de statiske feature-vektorene til MLP per definisjon kastet bort. At modusen til en drone manifesterer seg som et tidsmønster i RF-signalet, ikke bare som ett gjennomsnittlig spektrum, støttes av at nøyaktigheten steg til over 70 % allerede med en svært liten modell.

Modellkapasitet viste seg å være en avgjørende variabel. Det første forsøket med 203 461 parametere kollapset til én-klasse-prediksjon, analogt med Kitchen Sink-kollapsen for MLP. Med kun 181 treningsfiler er det matematisk sett et svært lavt antall parameteroppdateringer per treningsparameter, noe som betyr at en stor modell raskt memorerer støy fremfor å generalisere. Den endelige modellen på 11 000 parametere, kombinert med L2-straff på alle lag, dropout og GaussianNoise-augmentering, gav et trenings-/valideringsgap som læringsforløpet i @fig-cnn-curves bekrefter er lite. Dette viser at valg av riktig modellkapasitet, ikke bare regulariseringsteknikk, er det primære middelet mot overfitting ved små datasett.

Klasseimbalansen ble håndtert gjennom frekvensbaserte klassevekter, men resultatene viser at dette ikke var tilstrekkelig for alle klasser. Modus 2 og modus 3 oppnådde recall på henholdsvis 22 % og 25 %, mens modus 0, 1 og 4 ble klassifisert riktig. En sannsynlig forklaring er strukturen i BUI-koden: klasse 2 og 3 er definert av de to siste bitene og inneholder opptak fra to ulike dronefamilier, eksempelvis Bebop og AR Drone, i den samme klassen. Dersom RF-signaturen for «automatisk svev» varierer mellom produsenter, vil intra-klasse-variansen i klasse 2 og 3 være fundamentalt høyere enn i klasser dominert av én dronefamilie. Forvirringsmatrisen i @fig-cnn-confusion støtter dette: modus 2 og 3 forveksles ikke primært med hverandre, men fordeles mellom modus 0 og 4, noe som tyder på at modellen lærer produsentspesifikke signaturer og ikke en modusspesifikk signatur som er stabil på tvers av produsenter.

Funnet underbygger det samme argumentet som ble reist under valg av klasser: klassifisering av dronemodus på tvers av produsenter er en vanskeligere oppgave enn klassifisering av dronefamilie, fordi RF-protokollene som definerer en modus ikke nødvendigvis er felles for ulike fabrikanter. CNN-modellen mestrer oppgaven bedre enn MLP ved å utnytte temporal struktur, men støter på den samme grunnleggende utfordringen som MLP: høy intra-klasse-varians i de klassene som blander produsentfamilier.

==
