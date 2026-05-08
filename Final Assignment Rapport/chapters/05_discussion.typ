= Drøfting 

== Valg av klasser

Vi valgte å klassifisere dronemodus fremfor dronetype, fordi modus gir mer operasjonell informasjon — det skiller mellom en drone som er påslått, en som flyr, og en som sender video. I et reelt scenario er det gjerne mer nyttig å vite hva dronen gjør enn hvilken modell det er. Klassifisering av dronetype ville gitt et enklere problem, men med begrenset verdi: datasettet inneholder bare tre modeller, mens det i virkeligheten finnes langt flere — og det er ikke uvanlig å bygge droner selv.

En viktig konsekvens av dette valget er at klassene ikke er rent separate. BUI-koden bruker de to siste bitene til å kode modus på tvers av alle dronefamilier, men RF-protokollene som definerer en modus er produsentspesifikke. En Bebop-drone i «automatisk svev» og en AR Drone i «automatisk svev» bruker ikke nødvendigvis de samme frekvensene eller signalmønstrene. Å slå disse sammen i én klasse øker variasjonen innad i klassen og gjør klassifisering vanskeligere enn om modusene var separert per dronefamilie. Resultatene bekrefter dette: de to klassene med lavest ytelse er nettopp de to der BUI-koden samler to ulike dronefamilier.

En alternativ tilnærming ville vært å klassifisere dronefamilie og dronemodus separat, og kombinere prediksjonene i etterkant. Dette ville krevd mer data og en annen klasseinndeling, men kunne potensielt gitt bedre ytelse per modus.



== Prestasjon MLP

MLP-modellen oppnådde en testnøyaktighet på 82,61 % og en macro-F1 på 0,82. Den viktigste faktoren for dette resultatet er signalrepresentasjonen: ved å aggregere 300 overlappende vinduer per opptak til 192 egenskaper (gjennomsnitt, standardavvik og maksimum per frekvensbånd) får modellen tilgang til informasjon om hvordan signalet varierer gjennom opptaket. Standardavviket fanger om aktiviteten er stabil eller skiftende fra vindu til vindu, og maksimumverdien bevarer informasjon om toppenergi som et enkelt gjennomsnitt ville skjult.

Et interessant funn er at modellen med den sterkeste regulariseringen (α = 1,0) og den enkleste arkitekturen (ett lag med 64 nevroner) slo alle større alternativer. Med kun 181 treningsfiler vil en modell med for mange frihetsgrader memorere spesifikke opptak fremfor å generalisere. GridSearchCV med macro-F1 som mål bekreftet dette: søket favoriserte konsekvent de minste og mest regulariserte konfigurasjonene.

MLP presterer noe bedre enn CNN (82,6 % vs 71,7 %) til tross for at CNN behandler vinduesekvensen direkte og ikke komprimerer den til tre statistikker per bånd. En mulig forklaring er at gjennomsnitt, standardavvik og maksimum per frekvensbånd fanger opp det meste av den nyttige informasjonen for dette datasettet, og at mer detaljert strukturinformasjon ikke tilfører noe gitt det lille treningssettet. Begge modellene sliter med de samme klassene — modus 2 og 3 — noe som tyder på at problemet ligger i dataene, ikke i valg av modelltype.



== Prestasjon CNN

CNN-modellen oppnådde 71,7 % testnøyaktighet og macro-F1 på 0,66. At modellen klarer dette med kun 11 000 parametere tyder på at RF-signalet inneholder nyttig strukturinformasjon utover et enkelt gjennomsnittsspekter — vinduesekvensen bærer informasjon som de tre statistiske sammendragene til MLP delvis mister.

Modellkapasitet viste seg å være den avgjørende variabelen. Et første forsøk med 203 461 parametere kollapset til én-klasse-prediksjon, analogt med Kitchen Sink-kollapsen for MLP. Med kun 181 treningsfiler er antall unike treningspunkter per parameter svært lavt, noe som betyr at en stor modell raskt memorerer støy. Den endelige modellen på 11 000 parametere, kombinert med L2-straff, dropout og GaussianNoise-augmentering, gav et lite og stabilt gap mellom trenings- og valideringskurver, som vist i @fig-cnn-curves. Dette viser at riktig modellkapasitet er det primære virkemiddelet mot overfitting ved små datasett — ikke bare regulariseringsteknikk.

Modus 2 og 3 oppnådde recall på henholdsvis 22 % og 25 %, mens modus 0, 1 og 4 ble klassifisert med god eller perfekt recall. BUI-koden slår her sammen opptak fra to ulike dronefamilier i samme klasse — for eksempel Bebop og AR Drone under modus 10. Dersom RF-signaturen for «flyging uten video» varierer mellom produsenter, vil variasjonen innad i klassen være for høy til at modellen kan lære én felles signatur. Forvirringsmatrisen i @fig-cnn-confusion bekrefter dette: modus 2 og 3 forveksles ikke primært med hverandre, men fordeles mellom modus 0 og 4. Modellen lærer produsentspesifikke mønstre, ikke modusspesifikke. Begge modellene støter på det samme grunnleggende problemet, noe som peker mot at løsningen ligger i datastrukturen heller enn i modellvalget.

==
