= Konklusjon <Konklusjon>

== Konklusjon

Prosjektet undersøkte om maskinlæringsmodeller kan klassifisere dronemodus basert på RF-signaler fra DroneRF-datasettet. Resultatet er at begge modellene klarer oppgaven godt, men med ulike styrker og svakheter.

MLP-modellen oppnådde en testnøyaktighet på 82,61 % og en macro-F1 på 0,82. Den avgjørende forbedringen kom ikke fra arkitekturen, men fra en bedre signalrepresentasjon: ved å bytte fra én enkelt FFT per opptak til 192 statistisk aggregerte frekvensbånds-energier fikk modellen langt mer informasjon å arbeide med. Den tyngste regulariseringen (α = 1,0) og den enkleste arkitekturen (ett lag med 64 nevroner) vant konsekvent i hyperparameterletingen, noe som viser at datasettets størrelse setter et tak for nyttig modellkapasitet.

CNN-modellen oppnådde 71,7 % testnøyaktighet og macro-F1 på 0,66. Modellen klassifiserte tre av fem klasser godt, men slet med modus 2 og 3. Begge modellene feiler på de samme klassene, noe som peker mot en felles årsak: BUI-koden slår sammen opptak fra ulike dronefamilier i disse to klassene, noe som øker variasjonen innad i klassen og gjør dem vanskeligere å lære.

Prosjektet bekrefter at RF-signaler inneholder tilstrekkelig informasjon til å klassifisere dronemodus med god treffsikkerhet, og at valg av signalrepresentasjon er viktigere enn valg av modelltype for dette datasettet.

== Veien videre

De to klassene med lavest ytelse (modus 2 og 3) er de modusene der to ulike dronefamilier er samlet i én klasse. En naturlig forbedring ville være å dele klassene per dronefamilie, slik at modellen ikke forsøker å lære én felles signatur for signaler fra ulike produsenter. Dette ville kreve en annen klasseinndeling i BUI-koden.

En annen forbedring er å øke datasettets størrelse. Med 227 opptak og fem klasser er treningssettet knapt, og modellene er sterkt begrenset av dette. Flere opptak — særlig for modus 2 og 3 — ville trolig gi bedre og mer stabile resultater.

Til slutt er det verdt å undersøke om kryss-validering på filnivå (fremfor på enkeltopptak) gir et mer realistisk bilde av generaliseringsevnen, siden overlappende vinduer fra samme fil kan gjøre modellen bedre enn den faktisk er på nye opptak.




#pagebreak()

#align(center + horizon)[[Denne siden er blank med hensikt]]