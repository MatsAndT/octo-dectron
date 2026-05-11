= Drøfting 

== Valg av klasser

Vi valgte å klassifisere dronemodus fremfor dronetype, fordi modus gir mer operasjonell informasjon — det skiller mellom en drone som er påslått, en som flyr, og en som sender video. I et reelt scenario er det gjerne mer nyttig å vite hva dronen gjør enn hvilken modell det er. Klassifisering av dronetype kunne imidlertid gitt et enklere problem da det er å forvete signaler for moduser varierer fra produsent til produsent, men med det ville gitt begrenset verdi: datasettet inneholder bare tre modeller, mens det i virkeligheten finnes langt flere, og det er ikke uvanlig å bygge droner selv.

En viktig konsekvens av dette valget er at klassene ikke er rent separate. BUI-koden bruker de to siste bitene til å kode modus på tvers av alle dronefamilier, men RF-protokollene som definerer en modus er produsentspesifikke. En Bebop-drone i «automatisk _hover_» og en AR Drone i «automatisk _hover_» bruker ikke nødvendigvis de samme frekvensene eller signalmønstrene. Å slå disse sammen i én klasse øker variasjonen innad i klassen og gjør klassifisering vanskeligere enn om modusene var separert per dronefamilie ettersom det er å forvente mere likhet i frekvensspekteret innad hos en produsent.

En alternativ tilnærming ville vært å klassifisere dronefamilie og dronemodus separat, og kombinere prediksjonene i etterkant. Dette ville krevd mer data og en annen klasseinndeling, men kunne potensielt gitt bedre ytelse per modus.



== Prestasjon MLP

MLP-modellen oppnådde en testnøyaktighet på 82,61 % og en macro-F1 på 0,82. Den viktigste faktoren for dette resultatet er signalrepresentasjonen: ved å sammenstille 300 overlappende vinduer per opptak til 192 egenskaper (gjennomsnitt, standardavvik og maksimum per frekvensbånd) får modellen tilgang til informasjon om hvordan signalet varierer gjennom opptaket. Standardavviket fanger om aktiviteten er stabil eller skiftende fra vindu til vindu, og maksimumverdien bevarer informasjon om toppenergi som et enkelt gjennomsnitt ville skjult.

Et interessant funn er at modellen med den sterkeste regulariseringen (α = 1,0) og den enkleste arkitekturen (ett lag med 64 nevroner) slo alle større alternativer vi testet. Med kun 181 treningsfiler vil en modell med for mange frihetsgrader memorere spesifikke opptak fremfor å generalisere. GridSearchCV med macro-F1 som mål bekreftet dette: søket favoriserte konsekvent de minste og mest regulariserte konfigurasjonene.

MLP presterer noe bedre enn CNN (82,6 % vs 71,7 %) til tross for at CNN behandler vinduesekvensen direkte og ikke komprimerer den til tre statistikker per bånd. En mulig forklaring er at gjennomsnitt, standardavvik og maksimum per frekvensbånd fanger opp det meste av den nyttige informasjonen for dette datasettet. CNN ble tunet manuelt ved å se på lærings-kurvene, mens MLP ble søkt systematisk med GridSearchCV; det er mulig at CNN hadde hatt nytte av et tilsvarende systematisk søk.

I den tidlige fasen av prosjektet slet vi med at vår første versjon av MLP-modellen (v1) stagnerte på en treningsnøyaktighet rundt 40 % og en testnøyaktighet ned mot 20–30 %. Denne versjonen baserte seg på én enkelt fouriertransformasjon (FFT) for hele signalsekvensen, hvor vi kun hentet ut 28 statiske trekk (som de sterkeste frekvensene, median og ytterpunkter i lav- og høy-båndet). Eksplorativ dataanalyse (EDA) bekreftet mistanken vår: Problemet lå ikke i selve modellen eller hyperparameterne, men i et for tynt datagrunnlag med altfor liten variasjon. Vendepunktet kom da vi gikk bort fra å se på hele signalet som én blokk. Med råd fra faglige veiledere gikk vi over til å dele opp signalet i 300 mindre, overlappende vinduer, og kunne i stedet bygge en langt rikere profil. I den endelige modellen beregner vi statistikk (gjennomsnitt, standardavvik og maksimum) på tvers av alle disse vinduene. 

At vi beregner statistikk «på tvers av alle vinduene» betyr at vi analyserer hvordan hver enkelt frekvens utvikler seg gjennom hele opptaket. I stedet for at modellen må forholde seg til 300 separate «øyeblikksbilder», eller bare får én flat verdi for hele fila, ser vi på verdiene i hver frekvenskanal gjennom samtlige vinduer. Ved å regne ut gjennomsnitt, standardavvik og maksimum for hver kanal, oppsummerer vi signalets oppførsel: Gjennomsnittet viser den typiske profilen, standardavviket forteller oss om signalet er stabilt eller urolig, og maksimumsverdien sørger for at vi ikke mister de korte, men viktige energitoppene. Dette gir MLP-modellen en konsentrert «signatur» som inneholder informasjon om tidsutviklingen i signalet, uten at den drukner i rådata. Dette gir totalt 192 trekk som fanger opp både den typiske frekvensen og hvordan signalet endrer seg over tid. Denne overgangen fra statiske til tidsbaserte trekk var avgjørende for at modellen til slutt nådde en testnøyaktighet på 82,6 %.


== Prestasjon CNN

CNN-modellen oppnådde 71,7 % testnøyaktighet og macro-F1 på 0,66.

Modellkapasitet viste seg å være den avgjørende variabelen. Et første forsøk med 203 461 parametere kollapset til én-klasse-prediksjon. Med kun 181 treningsfiler er antall unike treningspunkter per parameter svært lavt, noe som betyr at en stor modell raskt memorerer støy. Den endelige modellen på 11 000 parametere, kombinert med L2-straff, dropout og GaussianNoise-augmentering, ga et lite og stabilt gap mellom trenings- og valideringskurver, som vist i @fig-cnn-curves.

Det er likevel grunn til å spørre om 11 000 parametere er for lavt — altså om modellen underfitter. Lærings-kurvene viser at trenings- og valideringstapet konvergerer tett og ikke divergerer, noe som vanligvis peker mer mot kapasitetsgrense enn grov underfitting. Modellen er med andre ord trolig nær grensen for hva 181 treningsfiler kan bære. For å undersøke dette systematisk ville det vært interessant å trene modeller med progressivt høyere parameterantal og følge validerings-F1 — men dette krever et større datasett for å gi meningsfulle estimater.

Modus 2 og 3 oppnådde recall på henholdsvis 22 % og 25 %, mens modus 0, 1 og 4 ble klassifisert med god eller perfekt recall. Dersom RF-signaturen for en modus, for eksempel «flyging uten video» varierer mellom produsenter, vil variasjonen innad i klassen være for høy til at modellen kan lære én felles signatur. Forvirringsmatrisen i @fig-cnn-confusion bekrefter dette: modus 2 og 3 forveksles ikke primært med hverandre, men fordeles mellom modus 0 og 4. Dette kan tyde på at modellen lærer produsentspesifikke mønstre, ikke modusspesifikke.

== Svar på forskningsspørsmålene

Sett opp mot de tre forskningsspørsmålene fra innledningen: 

1. *MLP på frekvens-statistikker:* Modellen nådde 82,61 % testnøyaktighet og macro-F1 0,82 — godt over baseline. Signalrepresentasjonen (192 sammensatte egenskaper) var den avgjørende faktoren, ikke arkitekturen.

2. *CNN på vinduesekvenser:* Modellen nådde 71,74 % testnøyaktighet og macro-F1 0,66 — klart over baseline, men svakere enn MLP. Kapasitetsbegrensningen (11k parametere) og mulig manglende likhet i RF-dataen for modusklassifisering på tvers av produsent er de viktigste forklaringene.

3. *Hvilke moduser forveksles, og hvorfor:* Modus 2 og 3 gir lavest ytelse i begge modeller. Feilmønsteret — at de forveksles med modus 0 og 4 snarere enn med hverandre — tyder på at det er mulig produksjonsspesifikke, ikke modusspesifikke RF-signaturer, og at dette er en datasettstruktur-svakhet eller klassifiseringsproblem (at det ikke går an å fullt ut generalisere modus på tvers av droneprodusenter) snarere enn en modellsvakhet.

== Kitchen Sink-modellens prestasjon

Kitchen Sink-modellen er en uregulert MLP med to skjulte lag (256 og 128 nevroner) og ingen L2-straff (α = 0). Den oppnår 100 % treningsnøyaktighet, noe som viser at den memorerer treningsdataene. Likevel ender den øverst i testnøyaktighet (84,78 %) og macro-F1 (0,85), over den regulariserte MLP-en som ble søkt systematisk med GridSearchCV.

Den viktigste forklaringen er at forskjellen er svært liten: 84,78 % av 46 testopptak tilsvarer 39 riktige, mens 82,61 % tilsvarer 38 riktige — én prøve. Med et testsett på 46 opptak er standardfeilen på en nøyaktighetsestimator stor nok til at de to modellene statistisk sett er likeverdige. En annen tilfeldig oppdeling av trenings- og testsett kunne like gjerne snu rekkefølgen.

En annen del av forklaringen ligger i hva som ble optimert. GridSearchCV søkte etter høyest mulig macro-F1, ikke høyest nøyaktighet. Macro-F1 vekter alle klasser likt, noe som betyr at GridSearchCV bevisst ofret litt nøyaktighet på majoritetsklassen for å forbedre recall på de sjeldnere klassene. Kitchen Sink ble aldri optimert mot noe som helst — den bare fikk frihet til å lære treningsdataene fullt ut.

En tredje faktor er at datasettet er samlet inn under kontrollerte forhold, og at train/test-splitten er stratifisert. Det betyr at trenings- og testsettet er svært like i sammensetning. Når treningsdata og testdata kommer fra de samme opptakssesjonene under de samme forholdene, er det ikke urimelig at en modell som har memorert treningsdataene likevel treffer godt på testsettet — fordi testdataene faktisk ligner treningsdataene mer enn de ville gjort i et reelt scenario med nye droner eller nye miljøer.

At Kitchen Sink-modellen kommer best ut, betyr ikke at memorering er en hensiktsmessig strategi. Resultatet skyldes nok heller et begrenset testsett.  Med en større og mer variert datamengde ville sannsynligvis den regulariserte modellen, basert på systematisk optimalisering, prestert bedre enn de andre modellene.