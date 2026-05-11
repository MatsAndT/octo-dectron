= Konklusjon <Konklusjon>

== Konklusjon

Prosjektet undersøkte om maskinlæringsmodeller kan klassifisere dronemodus basert på RF-signaler fra DroneRF-datasettet. Resultatet er at begge modellene klarer oppgaven godt, men med ulike styrker og svakheter.

MLP-modellen oppnådde en testnøyaktighet på 82,61 % og en macro-F1 på 0,82. Den avgjørende forbedringen kom ikke fra arkitekturen, men fra en bedre signalrepresentasjon: ved å bytte fra én enkelt FFT per opptak til 192 statistisk aggregerte frekvensbånds-energier fikk modellen langt mer informasjon å arbeide med. Den tyngste regulariseringen (α = 1,0) og den enkleste arkitekturen (ett lag med 64 nevroner) vant konsekvent i hyperparameterletingen, noe som viser at datasettets størrelse setter et tak for nyttig modellkapasitet.

CNN-modellen oppnådde en testnøyaktighet på 71,7 % og en macro-F1 på 0,66. Selv om modellen identifiserte tre av fem klasser med god presisjon, var resultatene for modus 2 og 3 svakere. En sannsynlig årsak er at RF-signaturene i disse klassene er mer preget av produsentspesifikke egenskaper enn av selve modusen. Dette skaper en stor variasjon innad i hver klasse, som gjør det utfordrende for modellener generelt å generalisere modus på tvers av ulike droneprodusenter.


Prosjektet bekrefter at RF-signaler inneholder tilstrekkelig informasjon til å klassifisere dronemodus med relativt god treffsikkerhet, og at valg av signalrepresentasjon for maskinlæringsmodellen er viktigere enn valg av klassifiseringstype (modus eller produsent) for dette datasettet.


== Veien videre

For å forbedre prestasjonen med å identifisere modus på tvers av ulike produsenter kunne en naturlig forbedring vært å dele klassene per droneprodusent og så lære modus for hver av disse produsentene, slik at modellen ikke forsøker å lære én felles signatur for signaler fra ulike produsenter. Dette ville kreve en annen klasseinndeling for maskinlæringsmodellen. Altså at man hadde trent opp på, og fått en predikasjon på for eksempel "Phantom _hover_". Dette fordrer imidlertid et større datasett for å kunne får tilstrekkelig data per modus per produsent. Med 227 opptak og fem klasser er treningssettet knapt, og modellene er sterkt begrenset av dette. Flere opptak — særlig for modus 2 og 3 — ville trolig gi bedre og mer stabile resultater.

En naturlig forbedring for å styrke identifikasjonen av operasjonsmodus på tvers av ulike produsenter ville vært å introdusere en mer finkornet klasseinndeling. Ved å trene modellen på produsentspesifikke modus, som for eksempel «Phantom_hover», kunne man redusert utfordringen med høy varians innad i klassene. Dette ville hindret at modellen tvinges til å utlede én felles signatur fra signaler som teknisk sett er svært ulike. Dette ville hindret at modellen tvinges til å utlede én felles signatur fra signaler som teknisk sett er divergerende. En slik strategi forutsetter imidlertid et betydelig større datasett for å sikre tilstrekkelig representativitet for hver enkelt kombinasjon av produsent og modus. Med kun 227 opptak er det nåværende datagrunnlaget for begrenset til å støtte en så detaljert oppsplitting, og en utvidelse av antall observasjoner – særlig for modus 2 og 3 – ville sannsynligvis gitt modellene den nødvendige statistiske styrken til å oppnå mer robuste resultater.



#pagebreak()

#align(center + horizon)[[Denne siden er blank med hensikt]]